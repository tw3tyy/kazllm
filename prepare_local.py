"""Download only missing allowed local weights; never refresh an existing tag."""
import os
import subprocess
from kazllm import query
from pipeline import load_inputs

if __name__ == "__main__":
    config, _, _ = load_inputs()
    installed = {m["name"] for m in query.request("/api/tags").get("models", [])}
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_NO_CLOUD="1")
    for spec in config["models"]:
        if spec.get("backend", "ollama_local") != "ollama_local":
            continue
        query.validate_model_name(spec["id"])
        if spec["id"] in installed:
            print("Already installed, keeping current weights: " + spec["id"])
        else:
            subprocess.run(["ollama", "pull", spec["id"]], env=env, check=True)
        query.model_identity(spec["id"])
