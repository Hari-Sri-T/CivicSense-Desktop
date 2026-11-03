import sys
import threading
from types import SimpleNamespace

import pytest

from model_runtime import ModelRuntime, parse_answer, MODEL_FILE, MODEL_REVISION


@pytest.mark.parametrize("raw", [
    '{"answer":"Yes","explanation":"Source 1"}',
    '```json\n{"answer":"Yes","explanation":"Source 1",}\n```',
    '<think>{"draft":"ignore"}</think>{"answer":"Yes","explanation":"Source 1"}',
])
def test_answer_repair(raw):
    assert parse_answer(raw) == {"answer": "Yes", "explanation": "Source 1"}


@pytest.mark.parametrize("raw", ['[]', '{"answer": "Yes"}', '{"answer": {}, "explanation":"x"}',
                                 '{"answer":"", "explanation":"x"}'])
def test_bad_answers_rejected(raw):
    with pytest.raises(ValueError):
        parse_answer(raw)


def test_prepare_runs_once_and_reuses_ready_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CIVICSENSE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CIVICSENSE_BACKEND", "embedded")
    runtime = ModelRuntime()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return "model.gguf"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=lambda **kwargs: object()))
    monkeypatch.setattr(runtime, "embedder", lambda: object())
    runtime.start()
    assert entered.wait(5)
    runtime.start()
    release.set()
    for thread in threading.enumerate():
        if thread.name == "civicsense-model-setup":
            thread.join(5)
    assert runtime.status()["phase"] == "ready"
    runtime.start()
    assert len(calls) == 1
    assert calls[0]["filename"] == MODEL_FILE
    assert calls[0]["revision"] == MODEL_REVISION


def test_setup_failure_reports_error(monkeypatch):
    monkeypatch.setenv("CIVICSENSE_BACKEND", "embedded")
    runtime = ModelRuntime()
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    runtime._prepare()
    assert runtime.status()["phase"] == "error"
    assert runtime.status()["error"]


def test_external_backend_uses_actual_model_and_follows_switches(monkeypatch):
    monkeypatch.setenv("CIVICSENSE_BACKEND", "lmstudio")
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
    runtime = ModelRuntime()
    available = [SimpleNamespace(id="qwen2.5-3b-instruct")]
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="response"))])

    runtime._client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=available)),
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    runtime._update(phase="ready")
    assert runtime.complete([]) == "response"
    assert calls[-1]["model"] == "qwen2.5-3b-instruct"
    available[:] = [SimpleNamespace(id="different-model")]
    runtime.complete([])
    assert calls[-1]["model"] == "different-model"
    monkeypatch.setenv("LM_STUDIO_MODEL", "missing-model")
    with pytest.raises(RuntimeError, match="not available"):
        runtime.complete([])
    monkeypatch.delenv("LM_STUDIO_MODEL")
    available.append(SimpleNamespace(id="ambiguous-model"))
    with pytest.raises(RuntimeError, match="Load one model"):
        runtime.complete([])
