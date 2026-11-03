"""Optional real-model smoke check. Downloads default weights when not cached."""
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app
from model_runtime import runtime


def main():
    runtime._prepare()
    state = runtime.status()
    if state["phase"] != "ready":
        raise SystemExit(f"Model preparation failed: {state.get('error')}")
    sample = Path(__file__).resolve().parents[1] / "examples/sample_policy.txt"
    with app.test_client() as client:
        response = client.post("/upload", data={"file": (io.BytesIO(sample.read_bytes()), sample.name)})
        assert response.status_code == 200, response.json
        observations = []
        for query in ["How much notice is required to cancel membership?", "What parking arrangements does the policy describe?"]:
            start = time.monotonic()
            response = client.post("/query", json={"query": query})
            assert response.status_code == 200, response.json
            data = response.json
            assert isinstance(data["answer"], str) and data["answer"].strip()
            assert isinstance(data["explanation"], str) and data["explanation"].strip()
            assert data["retrieval_details"]
            observations.append({"query": query, "answer": data["answer"], "explanation": data["explanation"],
                                 "latency_seconds": round(time.monotonic() - start, 2)})
        print(json.dumps({"backend": state["backend"], "model": state["model"], "observations": observations}, indent=2))
        client.post("/clear_index")


if __name__ == "__main__":
    main()
