"""Dispatch between local Ollama and the explicitly configured direct Gemini adapter. No simulated fallback."""
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = 'http://127.0.0.1:11434'
ALLOWED_MODELS = {'qwen2.5:1.5b', 'llama3.2:1b', 'gemma3:latest'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_model_name(name):
    if name not in ALLOWED_MODELS:
        raise ValueError('Only the three configured local model tags are allowed; cloud models are disabled')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Redirect refused: inference must remain on this computer')


def request(path, payload=None):
    if path not in ('/api/tags', '/api/show', '/api/chat', '/api/version', '/api/generate'):
        raise ValueError('Unsupported local endpoint')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request(BASE_URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with opener.open(req, timeout=600) as r:
            result = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Local Ollama HTTP {exc.code}; no substitute answer generated') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('Cannot reach local Ollama. Start it using start_ollama_local.sh') from None
    if result.get('error'):
        raise RuntimeError('Local model returned an error; no substitute answer generated')
    return result


def model_identity(model):
    validate_model_name(model)
    matches = [m for m in request('/api/tags').get('models', []) if m.get('name') == model]
    if len(matches) != 1 or not matches[0].get('digest'):
        raise ValueError('Local weights not installed: ' + model + '. Run prepare_local.sh')
    show = request('/api/show', {'model': model})
    if show.get('remote_host') or show.get('remote_model'):
        raise ValueError('Cloud-backed model refused')
    return {'name': model, 'digest': matches[0]['digest'], 'details': matches[0].get('details'),
            'show_sha256': digest(show), 'show': show}


def make_payload(row, spec):
    if spec.get("backend") == "gemini":
        from .gemini_query import make_payload as gemini_payload
        return gemini_payload(row, spec)
    validate_model_name(spec['id'])
    budget = spec['qa_max_tokens'] if row['task'] == 'qa' else spec['mt_max_tokens']
    if len(row['prompt'].encode('utf-8')) + budget + 512 > spec['num_ctx']:
        raise ValueError('Prompt exceeds conservative context bound: ' + row['prompt_id'])
    return {'model': spec['id'], 'messages': [{'role': 'user', 'content': row['prompt']}],
            'stream': False, 'keep_alive': '5m',
            'options': {'temperature': spec['temperature'], 'seed': spec['seed'],
                        'num_ctx': spec['num_ctx'], 'num_predict': budget}}


def validate_response(response, expected_model=None):
    if response.get('done') is not True or response.get('done_reason') != 'stop':
        raise ValueError('Incomplete/truncated local response rejected')
    if expected_model and response.get('model') != expected_model:
        raise ValueError('Unexpected returned model')
    if not isinstance(response.get('model'), str) or not response['model']:
        raise ValueError('Missing model name')
    text = response.get('message', {}).get('content')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty local answer rejected')
    return text


def read_validated(path, prompts, spec, identity=None):
    if spec.get("backend") == "gemini":
        from .gemini_query import read_validated as gemini_read
        return gemini_read(path, prompts, spec)
    path = Path(path)
    if not path.exists():
        return []
    expected = {p['prompt_id']: p for p in prompts}
    seen, rows, identities = set(), [], set()
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        pid = r['prompt_id']
        if pid in seen or pid not in expected:
            raise ValueError('Duplicate or unknown prompt ID')
        if r.get('provenance') != 'ollama_local' or r.get('requested_model') != spec['id']:
            raise ValueError('Old, cloud, or unverified outputs refused')
        if r.get('prompt_sha256') != digest(expected[pid]['prompt']):
            raise ValueError('Prompt changed')
        if r.get('request_sha256') != digest(make_payload(expected[pid], spec)):
            raise ValueError('Decoding settings changed')
        if r.get('output') != validate_response(r['response'], spec['id']) or r.get('model') != spec['id']:
            raise ValueError('Saved output/metadata does not match local response')
        if r.get('response_id') != digest(r['response']):
            raise ValueError('Saved local response checksum mismatch')
        fingerprint = r.get('weights_digest')
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('Missing local weights digest')
        identities.add((fingerprint, r.get('show_sha256'), r.get('ollama_version')))
        if identity and (fingerprint != identity['digest'] or r.get('show_sha256') != identity['show_sha256']):
            raise ValueError('Local model weights or template changed; do not mix experiments')
        if not r.get('show_sha256') or not r.get('ollama_version'):
            raise ValueError('Missing local model/runtime identity')
        elapsed = r.get('elapsed_seconds')
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid request duration')
        if datetime.fromisoformat(r['time']) < datetime.fromisoformat(r['started_at']):
            raise ValueError('Invalid timestamps')
        seen.add(pid)
        rows.append(r)
    if len(identities) > 1:
        raise ValueError('Mixed local model/runtime identities in one experiment')
    return rows


def run_model(prompts, out_path, spec, identity=None, version=None):
    if spec.get("backend") == "gemini":
        from .gemini_query import run_model as gemini_run
        return gemini_run(prompts, out_path, spec)
    path = Path(out_path)
    existing = read_validated(path, prompts, spec, identity)
    if existing and existing[0]['ollama_version'] != version:
        raise ValueError('Ollama version changed since the saved run')
    done = {r['prompt_id'] for r in existing}
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in prompts:
        make_payload(row, spec)
    with path.open('a', encoding='utf-8') as f:
        for row in prompts:
            if row['prompt_id'] in done:
                continue
            payload = make_payload(row, spec)
            started, tick = now(), time.perf_counter()
            response = request('/api/chat', payload)
            text = validate_response(response, spec['id'])
            record = {'prompt_id': row['prompt_id'], 'output': text, 'provenance': 'ollama_local',
                'model': spec['id'], 'requested_model': spec['id'], 'weights_digest': identity['digest'],
                'show_sha256': identity['show_sha256'], 'ollama_version': version,
                'response_id': digest(response), 'response_id_note': 'Local checksum, not a provider-issued request ID',
                'prompt_sha256': digest(row['prompt']), 'request_sha256': digest(payload),
                'started_at': started, 'time': now(), 'elapsed_seconds': time.perf_counter() - tick,
                'response': response}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            done.add(row['prompt_id'])
            print(f"{spec['name']}: {len(done)}/{len(prompts)}", flush=True)
    return len(done)
