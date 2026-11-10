"""Direct Gemini adapter. No automatic billing changes, model fallback or quota retry."""
import json
import math
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from .query import digest, now

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-3.5-flash-lite"


class QuotaExceeded(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Gemini redirect refused; credential not forwarded")


def validate_model_name(model):
    if model != MODEL:
        raise ValueError("Only the explicitly configured free-tier-eligible Gemini model is allowed")


def _read_response(opener, req):
    delays = (8, 16)
    for attempt in range(len(delays) + 1):
        try:
            with opener.open(req, timeout=180) as response:
                return json.load(response)
        except (ConnectionResetError, urllib.error.URLError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if not isinstance(reason, ConnectionResetError) or attempt == len(delays):
                raise
            delay = delays[attempt]
            print(f"Connection reset. Retrying the same request in {delay} seconds.", flush=True)
            time.sleep(delay)


def request(model, payload=None):
    validate_model_name(model)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing; enter it privately using run_gemini.sh")
    suffix = ":generateContent" if payload is not None else ""
    req = urllib.request.Request(BASE_URL + "/models/" + model + suffix,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        import certifi
    except ImportError:
        raise RuntimeError("Install certificate bundle: python3 -m pip install -r requirements.txt") from None
    context = ssl.create_default_context()
    # Some macOS Python installs have no system CA bundle configured. Add current
    # public roots while retaining the platform's existing trust configuration.
    context.load_verify_locations(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), NoRedirect())
    try:
        result = _read_response(opener, req)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise QuotaExceeded("Gemini quota exhausted (HTTP 429). Saved progress kept. Check AI Studio quota; resume later. No automatic retry or paid fallback.") from None
        if exc.code == 404:
            reason = "Generation endpoint/model unavailable for this project. A successful metadata GET does not prove generation access."
            try:
                message = json.loads(exc.read(32768)).get("error", {}).get("message", "").lower()
                if "new users" in message or "previously used" in message or "prior usage" in message:
                    reason = "Google restricts this legacy model to projects with prior usage; metadata can still be visible."
            except Exception:
                pass
            raise RuntimeError("Gemini HTTP 404. " + reason + " No automatic model or tariff replacement.") from None
        raise RuntimeError(f"Gemini HTTP {exc.code}; no substitute output. Check key, model access and region.") from None
    except (ConnectionResetError, urllib.error.URLError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, ConnectionResetError):
            raise RuntimeError("Connection to Google reset after two retries. Saved progress kept. Run the same command to continue.") from None
        if isinstance(reason, ssl.SSLCertVerificationError):
            detail = "HTTPS certificate verification failed; update certifi and check the network certificate configuration."
        elif isinstance(reason, socket.gaierror):
            detail = "Google server name could not be resolved; check the internet connection and DNS."
        elif isinstance(reason, (TimeoutError, socket.timeout)):
            detail = "Connection to Google timed out."
        else:
            detail = "Network connection to Google failed (" + type(reason).__name__ + ")."
        raise RuntimeError(detail + " Saved progress kept; no automatic retries.") from None
    except TimeoutError:
        raise RuntimeError("Connection to Google timed out; saved progress kept.") from None
    if result.get("error"):
        raise RuntimeError("Gemini returned an error; no answer generated locally")
    return result


def make_payload(row, spec):
    validate_model_name(spec["id"])
    return {"contents": [{"role": "user", "parts": [{"text": row["prompt"]}]}],
            "generationConfig": {"temperature": spec["temperature"], "seed": spec["seed"],
                "maxOutputTokens": spec["qa_max_tokens"] if row["task"] == "qa" else spec["mt_max_tokens"],
                "thinkingConfig": {"thinkingLevel": "MINIMAL"}}}


def validate_response(response):
    if response.get("promptFeedback", {}).get("blockReason"):
        raise ValueError("Gemini prompt blocked; no successful answer saved")
    candidates = response.get("candidates") or []
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Gemini incomplete, filtered or truncated response rejected")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought"))
    if not text.strip():
        raise ValueError("Gemini returned no answer text")
    if not response.get("responseId") or not response.get("modelVersion"):
        raise ValueError("Gemini response identity is missing")
    return text


def read_validated(path, prompts, spec):
    path = Path(path)
    if not path.exists():
        return []
    expected = {r["prompt_id"]: r for r in prompts}
    seen, rows, versions = set(), [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        pid = r["prompt_id"]
        if pid not in expected or pid in seen:
            raise ValueError("Duplicate/unknown Gemini prompt ID")
        if r.get("provenance") != "gemini_api" or r.get("requested_model") != spec["id"]:
            raise ValueError("Unverified or mismatched Gemini outputs")
        if r.get("prompt_sha256") != digest(expected[pid]["prompt"]) or r.get("request_sha256") != digest(make_payload(expected[pid], spec)):
            raise ValueError("Gemini prompt or settings changed")
        if r.get("output") != validate_response(r["response"]):
            raise ValueError("Gemini answer does not match raw response")
        if r.get("model") != r["response"]["modelVersion"] or r.get("response_id") != r["response"]["responseId"]:
            raise ValueError("Gemini metadata mismatch")
        elapsed = r.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid Gemini duration")
        if datetime.fromisoformat(r["time"]) < datetime.fromisoformat(r["started_at"]):
            raise ValueError("Invalid Gemini timestamps")
        if r.get("billing_verification") != "user_confirmed_free_project_not_verified_by_api":
            raise ValueError("Missing free-project declaration")
        versions.add(r["model"])
        seen.add(pid)
        rows.append(r)
    if len(versions) > 1:
        raise ValueError("Mixed Gemini model versions; do not combine them as one frozen run")
    return rows


def run_model(prompts, out_path, spec):
    if os.environ.get("GEMINI_FREE_TIER_CONFIRMED") != "1":
        raise RuntimeError("First verify Free tier with billing disabled in AI Studio. The API cannot prove billing status.")
    existing = read_validated(out_path, prompts, spec)
    done = {r["prompt_id"] for r in existing}
    version = existing[0]["model"] if existing else None
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    interval = spec.get("min_interval_seconds", 7)
    if not isinstance(interval, (float, int)) or interval < 1:
        raise ValueError("Gemini interval must be at least one second")
    for r in prompts:
        make_payload(r, spec)
    # Pause before every request, including the first request after resume.
    with path.open("a", encoding="utf-8") as f:
        for row in prompts:
            if row["prompt_id"] in done:
                continue
            time.sleep(interval)
            started, tick = now(), time.perf_counter()
            payload = make_payload(row, spec)
            response = request(spec["id"], payload)
            text = validate_response(response)
            if version and response["modelVersion"] != version:
                raise ValueError("Gemini serving version changed; stopped before mixing saved versions")
            version = response["modelVersion"]
            record = {"prompt_id": row["prompt_id"], "output": text, "provenance": "gemini_api",
                "requested_model": spec["id"], "model": version, "response_id": response["responseId"],
                "prompt_sha256": digest(row["prompt"]), "request_sha256": digest(payload),
                "started_at": started, "time": now(), "elapsed_seconds": time.perf_counter() - tick,
                "billing_verification": "user_confirmed_free_project_not_verified_by_api", "response": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done.add(row["prompt_id"])
            print(f"{spec['name']}: {len(done)}/{len(prompts)}", flush=True)
    return len(done)
