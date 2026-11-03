"""CivicSense's local model lifecycle. Importing this module never downloads weights."""

import json
import os
import sys
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse

MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_REVISION = "7dabda4d13d513e3e842b20f0d435c732f172cbe"
MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_NAME = "Qwen2.5 · 3B Instruct · Q4_K_M"
EMBED_REPO = "sentence-transformers/all-MiniLM-L6-v2"


def data_directory():
    override = os.getenv("CIVICSENSE_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "CivicSense"


class ModelRuntime:
    def __init__(self):
        self.backend = os.getenv("CIVICSENSE_BACKEND", "embedded")
        if self.backend not in {"embedded", "lmstudio"}:
            raise ValueError("CIVICSENSE_BACKEND must be embedded or lmstudio")
        self.directory = data_directory()
        self._state_lock = Lock()
        self._inference_lock = Lock()
        self._embedding_lock = Lock()
        self._model = None
        self._embedder = None
        self._client = None
        self._state = {"phase": "idle", "message": "Preparing your local workspace", "error": None}

    def status(self):
        with self._state_lock:
            return dict(self._state, backend=self.backend, model=MODEL_NAME,
                        model_repository=MODEL_REPO, data_directory=str(self.directory))

    def _update(self, **values):
        with self._state_lock:
            self._state.update(values)

    def start(self):
        with self._state_lock:
            if self._state["phase"] in {"starting", "downloading", "loading", "ready"}:
                return
            self._state.update(phase="starting", message="Checking local models", error=None)
        Thread(target=self._prepare, name="civicsense-model-setup", daemon=True).start()

    def embedder(self):
        with self._embedding_lock:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                existing = Path(__file__).resolve().parent / "all-MiniLM-L6-v2"
                # Reuse the original checkout's weights when present.
                local = existing / "model.safetensors"
                source = str(existing) if local.exists() and local.stat().st_size > 1000000 else EMBED_REPO
                self._embedder = SentenceTransformer(
                    source, device="cpu", cache_folder=str(self.directory / "embeddings")
                )
            return self._embedder

    def _prepare(self):
        try:
            if self.backend == "embedded":
                # Fail before a multi-GB download if the inference engine is missing.
                from llama_cpp import Llama
                from huggingface_hub import hf_hub_download
                from tqdm.auto import tqdm

                runtime = self

                class DownloadProgress(tqdm):
                    def update(self, amount=1):
                        result = super().update(amount)
                        if self.total:
                            runtime._update(message=f"Downloading Qwen weights · {min(100, self.n / self.total * 100):.0f}%")
                        return result

                self._update(phase="downloading", message="Downloading Qwen weights (about 2.1 GB); cached files are reused")
                model_path = hf_hub_download(
                    repo_id=MODEL_REPO, filename=MODEL_FILE, revision=MODEL_REVISION,
                    cache_dir=str(self.directory / "models"), tqdm_class=DownloadProgress,
                )
                self._update(phase="loading", message="Loading Qwen into memory")
                self._model = Llama(
                    model_path=model_path, n_ctx=8192, n_gpu_layers=0,
                    n_threads=max(1, (os.cpu_count() or 2) // 2), verbose=False,
                )
            else:
                from openai import OpenAI

                base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
                parsed = urlparse(base_url)
                if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                    raise ValueError("LM Studio URL must point to a local HTTP server")
                self._client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=180, max_retries=0)
                self._active_external_model()
            self._update(phase="loading", message="Preparing MiniLM embeddings (downloaded once)")
            self.embedder()
            self._update(phase="ready", message="Local models ready · your documents stay on this device")
        except Exception as exc:
            self._model = None
            self._update(phase="error", message="Model setup needs attention", error=str(exc))

    def _active_external_model(self):
        # Resolve on every call: research scripts switch loaded models between runs.
        models = self._client.models.list().data
        ids = [model.id for model in models if "embed" not in model.id.lower()]
        configured = os.getenv("LM_STUDIO_MODEL", "").strip()
        if configured and configured not in ids:
            raise RuntimeError(f"Configured model {configured!r} is not available in LM Studio")
        if not configured and len(ids) != 1:
            raise RuntimeError("Load one model in LM Studio, or set LM_STUDIO_MODEL to an available model ID")
        selected = configured or ids[0]
        self._update(model_id=selected)
        return selected

    def complete(self, messages):
        if self.status()["phase"] != "ready":
            self.start()
            raise RuntimeError("Local models are still preparing; check Models for setup status")
        with self._inference_lock:
            if self.backend == "lmstudio":
                response = self._client.chat.completions.create(
                    model=self._active_external_model(), messages=messages,
                    temperature=0.1, max_tokens=4096,
                )
                return response.choices[0].message.content or ""
            response = self._model.create_chat_completion(
                messages=messages, temperature=0.1, max_tokens=4096,
            )
            return response["choices"][0]["message"]["content"] or ""


def parse_answer(raw):
    """Repair JSON formatting, but reject data that is not an answer/explanation pair."""
    from json_repair import repair_json

    # Ignore any preamble, including JSON examples inside a reasoning block.
    raw = raw.rsplit("</think>", 1)[-1].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model returned no JSON answer")
    raw = raw[start:end + 1]
    try:
        result = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        result = repair_json(raw, return_objects=True)
    if not isinstance(result, dict) or any(
        not isinstance(result.get(key), str) or not result[key].strip()
        for key in ("answer", "explanation")
    ):
        raise ValueError("Model response must contain nonempty answer and explanation strings")
    return {key: result[key] for key in ("answer", "explanation")}


runtime = ModelRuntime()
