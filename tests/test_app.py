import io

import numpy as np
import pytest

import app as backend


class FakeEncoder:
    def encode(self, texts, convert_to_numpy=True):
        return np.array([[float("cancel" in text.lower()), 1.0] for text in texts])


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(backend.runtime, "embedder", lambda: FakeEncoder())
    monkeypatch.setattr(backend.runtime, "status", lambda: {"phase": "ready", "backend": "embedded"})
    monkeypatch.setattr(backend.runtime, "complete", lambda messages: '{"answer":"Seven days.","explanation":"Source 1 requires seven days notice."}')
    backend.app.config.update(TESTING=True, EVALUATION_ROOT=tmp_path)
    with backend.app.test_client() as client:
        client.post("/clear_index")
        yield client
        client.post("/clear_index")


def upload(client, text=b"Cancellation requires seven days notice.", name="policy.txt"):
    return client.post("/upload", data={"file": (io.BytesIO(text), name)})


def test_document_question_and_evidence(client):
    assert client.post("/query", json={"query": "Cancellation?"}).status_code == 409
    response = upload(client)
    assert response.status_code == 200
    assert response.json["document"]["chunk_count"] == 1
    response = client.post("/query", json={"query": "How do I cancel?"})
    assert response.status_code == 200
    assert response.json["answer"] == "Seven days."
    evidence = response.json["retrieval_details"][0]
    assert "seven days" in evidence["text"]
    assert evidence["file_path"] == "policy.txt"
    assert evidence["para_idx"] == 0
    assert response.json["top_k_used"] == 5
    client.post("/clear_index")
    assert client.get("/api/status").json["document"] is None


def test_failed_upload_preserves_current_document(client, monkeypatch):
    assert upload(client).status_code == 200
    assert upload(client, b"", "empty.txt").status_code == 400
    assert upload(client, b"anything", "script.exe").status_code == 400
    monkeypatch.setattr(backend.runtime, "embedder", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert upload(client, b"A second document.", "second.txt").status_code == 500
    assert client.get("/api/status").json["document"]["name"] == "policy.txt"


@pytest.mark.parametrize("payload", [None, [], {}, {"query": []}, {"query": "x", "top_k": 0},
    {"query": "x", "top_k": True}, {"query": "x", "top_k": "5"}, {"query": "x", "no_reasoning": "false"}])
def test_invalid_query(client, payload):
    assert client.post("/query", json=payload).status_code == 400


def test_retrieval_only_does_not_call_generator(client, monkeypatch):
    upload(client)
    monkeypatch.setattr(backend.runtime, "complete", lambda _: pytest.fail("generation called"))
    response = client.post("/query", json={"query": "cancel", "retrieval_only": True})
    assert response.status_code == 200
    assert response.json["answer"] is None
    assert response.json["retrieval_details"]


def test_invalid_model_output_is_an_error(client, monkeypatch):
    upload(client)
    monkeypatch.setattr(backend.runtime, "complete", lambda _: '["wrong"]')
    response = client.post("/query", json={"query": "cancel"})
    assert response.status_code == 502
    assert response.json["answer"].startswith("Error:")


def test_corpus_path_boundary(client, tmp_path):
    inside = tmp_path / "example.txt"
    inside.write_text("A valid document.")
    assert client.post("/load_file", json={"file_path": str(inside)}).status_code == 200
    assert client.post("/load_file", json={"file_path": "/etc/passwd"}).status_code == 403
    link = tmp_path / "escape.txt"
    try:
        link.symlink_to("/etc/passwd")
    except (OSError, NotImplementedError):
        return
    assert client.post("/load_file", json={"file_path": str(link)}).status_code == 403


def test_local_browser_boundaries(client):
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/clear_index", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/clear_index", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    response = client.get("/")
    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert b"CivicSense" in response.data


def test_chunk_boundaries_and_retrieval():
    parts = [{"text": "one two three", "page": 1}, {"text": "four five six seven", "page": 2}]
    chunks = backend.semantic_chunker(parts, "example.pdf", max_tokens=4, overlap_tokens=1)
    assert [c["text"] for c in chunks] == ["one two three four", "four five six seven", "seven"]
    assert chunks[1]["metadata"]["page"] == 2
    with pytest.raises(ValueError):
        backend.semantic_chunker(parts, "example.pdf", max_tokens=4, overlap_tokens=4)
    backend.store_in_memory([[1, 0], [0, 1]], chunks[:2])
    assert backend.retrieve_from_memory([0, 1], 1)[0][0]["text"] == chunks[1]["text"]
    backend.store_in_memory([], [])
    assert backend.retrieve_from_memory([0, 1]) == []


def test_failed_index_replacement_preserves_vectors(client):
    upload(client)
    with pytest.raises(ValueError):
        backend.store_in_memory([[1, 0]], [])
    with pytest.raises(ValueError):
        backend.store_in_memory([[float("nan"), 0]], [{"text": "bad"}])
    assert backend.document_chunk_store[0]["text"] == "Cancellation requires seven days notice."
