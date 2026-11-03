"""Diagnostics used to verify source and frozen desktop packages."""
import json
import tempfile
from pathlib import Path


def check_desktop(with_models=False):
    import numpy as np
    import pymupdf
    from llama_cpp import Llama
    from sentence_transformers import SentenceTransformer
    from app import app, extract_pdf_with_structure
    from model_runtime import parse_answer, runtime

    root = Path(__file__).resolve().parent
    for relative in ("templates/index.html", "static/app.css", "static/app.js", "static/civicsense-logo.png"):
        if not (root / relative).is_file():
            raise RuntimeError(f"Missing desktop resource: {relative}")
    with tempfile.TemporaryDirectory(prefix="civicsense-check-") as directory:
        path = Path(directory) / "sample.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((40, 60), "Membership can be cancelled with seven days notice.")
            document.save(path)
        assert "seven days" in extract_pdf_with_structure(path)[0]["text"]
    assert parse_answer('{"answer":"Ready","explanation":"Desktop check"}')["answer"] == "Ready"
    with app.test_client() as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/civicsense-logo.png").status_code == 200
    report = {"status": "passed", "resources": True, "pdf_parser": True,
              "generation_binding": Llama.__name__, "embedding_binding": SentenceTransformer.__name__,
              "models_checked": False}
    if with_models:
        runtime._prepare()
        state = runtime.status()
        if state["phase"] != "ready":
            raise RuntimeError(state.get("error") or "Models did not become ready")
        vectors = runtime.embedder().encode(["seven days notice"], convert_to_numpy=True)
        assert np.asarray(vectors).shape == (1, 384)
        with app.test_client() as client:
            sample = root / "examples/sample_policy.txt"
            with sample.open("rb") as stream:
                response = client.post("/upload", data={"file": (stream, sample.name)})
            assert response.status_code == 200, response.json
            response = client.post("/query", json={"query": "How much notice is needed to cancel membership?"})
            assert response.status_code == 200, response.json
            assert response.json["retrieval_details"]
            report["model_answer"] = response.json["answer"]
            client.post("/clear_index")
        report["models_checked"] = True
    return report


def write_desktop_check(path, with_models=False):
    try:
        report = check_desktop(with_models)
    except Exception as exc:
        Path(path).write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
        raise
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
