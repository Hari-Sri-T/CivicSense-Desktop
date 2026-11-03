# Project map

The repository is rooted in the original `CivicSense/` source directory. The parent workspace also contained manuscript versions, handover notes and submission packages. The relevant manuscript and review notes are included under `paper/`; virtual environments, model weights, secret files, uploads and duplicate ZIP bundles remain local.

| Area | Contents and role |
|---|---|
| `desktop.py` | Native window / optional browser launcher; starts loopback server and model preparation |
| `app.py` | Document parsing, 350/50 word chunking, MiniLM embeddings, in-memory retrieval, prompts and HTTP endpoints |
| `model_runtime.py` | Pinned Qwen GGUF download, llama.cpp lifecycle, optional LM Studio client, JSON validation |
| `templates/`, `static/` | Offline desktop interface, styling and client behavior; no CDN assets |
| `tests/` | Deterministic API, model lifecycle and parsing checks |
| `scripts/` | Release file scanning and executable build |
| `.github/workflows/` | Cross-platform unit checks and manually triggered desktop build |
| `examples/` | Fictional sample document suitable for a demo |
| `evaluate*.py` | Multiple generations of the research pipeline; see RESEARCH.md |
| `model_results/` | Saved per-model evaluation artifacts, including interim backups |
| `evaluation_results_*.json`, `*.csv` | Earlier full evaluation and judge artifacts; distinct from the final 500-query run |
| `ablation*`, `compare_embeddings.py`, `robustness*`, `scalability*` | Component experiments and saved results |
| `hallucination_checker.py`, `calc_metrics.py`, `compile_v2.py`, `count_valid.py`, `validate_results.py` | Scoring, aggregation and inspection utilities |
| `gen_teleai.py`, `teleai_attacks.json`, `jbb_behaviors.json` | Historical adversarial test generation/data; provenance limitations in evidence audit |
| `previous_codes/` | Earlier model-comparison scripts and results, retained for traceability |
| `nlpproject/` | Separate legacy sustainability sentiment experiment; not used by the desktop app |
| `update_latex.py`, `plot_chart.py` | Historical manuscript-patching and figure scripts; inspect paths before use |
| `paper/` | Supplied manuscript, figures, evaluation handover/review notes |
| `docs/` | Setup, architecture/research map, evidence audit, conference outline |

The original README described Gemini and manual embedding-model cloning. The active code had moved to LM Studio, loaded embeddings eagerly from a relative directory, and sent `model="local-model"` despite a configured model ID. The desktop release replaces that startup path with lazy local preparation and an explicit inference lifecycle while retaining the research API and prompt variants.

No model weights or Python environment are source code dependencies to be committed. A fresh install gets Python dependencies through the requirements files and model weights through the publisher's model hub. Existing local MiniLM weights can be reused when the original checkout is present.
