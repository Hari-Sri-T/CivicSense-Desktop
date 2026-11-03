# Research reproduction

Run commands from the repository root with an activated Python environment.

```bash
python -m pip install -r requirements-research.txt
```

The app itself does not need LegalBench-RAG. For experiments, obtain the corpus and benchmarks from [the upstream LegalBench-RAG repository](https://github.com/zeroentropy-ai/legalbenchrag), following its current dataset download instructions and licenses. Arrange the data as:

```text
LegalBench-RAG/
  corpus/{contractnli,cuad,maud,privacy_qa}/...
  benchmarks/{contractnli,cuad,maud,privacy_qa}.json
```

The original experiments switch models with the `lms` CLI. Install LM Studio, enable its local server at `127.0.0.1:1234`, download the required comparison models and make `lms` available on PATH. Launch the research backend in a separate terminal:

```bash
# Linux / macOS
CIVICSENSE_BACKEND=lmstudio python app.py
```

On PowerShell use `$env:CIVICSENSE_BACKEND='lmstudio'` followed by `python app.py`. Leave `LM_STUDIO_MODEL` unset so the app follows the model loaded by each experiment; a stale value in an existing `.env` must be removed for multi-model experiments. `/load_file` is limited to the corpus directory by default. `app.py` listens on port 5001 without debug mode.

| Script | Purpose |
|---|---|
| `compare_embeddings.py` | MiniLM / MPNet / DistilRoBERTa comparison |
| `ablation_chunk_size_fast.py` | Retrieval-only chunk and overlap sweep |
| `ablation_chunk_size.py` | Earlier chunk sweep with generation metrics |
| `evaluate_v2.py` | 24-query baseline across the comparison models |
| `evaluate_v2_phase2.py` | Restricted-prompt comparison |
| `evaluate_v3_phase3.py` | 500-query Qwen/Gemma evaluation |
| `evaluate_v4_security.py` | Historical adversarial query / judge experiment |
| `hallucination_checker.py` | Judge saved baseline responses against gold text |
| `evaluate1.py`, `evaluate2.py`, `evaluate3.py` | Earlier generation, metric, judge pipeline |
| `robustness_test.py`, `scalability_bench.py` | Exploratory robustness/resource checks |
| `compile_v2.py`, `calc_metrics.py`, `validate_results.py` | Inspect/aggregate saved outputs |

**Run experiments deliberately.** Several scripts overwrite their result files, switch/unload LM Studio models or download additional scoring models. Preserve the historical outputs in Git and save fresh experimental results separately. The old judge script edits result files in place. `gen_teleai.py` generates synthetic prompts and overwrites `teleai_attacks.json`; do not run it as if it fetches an authenticated TeleAI benchmark.

Example runs:

```bash
python evaluate_v2.py --models qwen2.5-3b-instruct
python evaluate_v3_phase3.py
python ablation_chunk_size_fast.py
```

`run_all.sh` is a convenience wrapper for the older exploratory pipeline, not an exact reproduction of every paper table. Set `CIVICSENSE_PYTHON` to override its interpreter. Scripts under `previous_codes/` are preserved historical utilities and may use old paths or schemas. `nlpproject/` contains a separate legacy sustainability-classification experiment, not a component of CivicSense Desktop; it additionally requires pandas, ollama, and the legacy google-generativeai package, plus its separately obtained model/data. Its cloud key now comes from `GOOGLE_API_KEY` instead of source code.

Read [EVIDENCE_AUDIT.md](EVIDENCE_AUDIT.md) before interpreting the saved metrics. Research scripts and saved outputs alone do not establish that every manuscript claim is reproducible.
