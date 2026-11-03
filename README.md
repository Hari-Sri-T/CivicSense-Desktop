<p align="center"><img src="static/icon.svg" width="76" alt="CivicSense icon"></p>
<h1 align="center">CivicSense Desktop</h1>
<p align="center">Your documents, understood. A local desktop companion for civic and legal documents.</p>

CivicSense turns PDF, DOCX and TXT documents into a searchable workspace. Ask a question, read a plain-language answer, and inspect the retrieved source passages alongside it. The desktop workflow takes inspiration from LM Studio: a local model, visible setup status, and a private document chat.

The default engine downloads **Qwen2.5-3B-Instruct Q4_K_M** and **all-MiniLM-L6-v2** automatically, then runs them on your computer. LM Studio and cloud API keys are not required. Qwen was selected for operational stability in the project's evaluation; this is not a claim that it is the best model on every benchmark.

![CivicSense desktop workspace](docs/desktop-preview.png)

## Run the desktop app

Use **64-bit Python 3.12**. Start with a modern desktop and at least 8 GB RAM; 16 GB is preferable. Allow several GB of free disk space for dependencies and model caches. Performance varies with hardware. The default CPU engine needs no dedicated GPU.

```bash
git clone https://github.com/Hari-Sri-T/CivicSense-Desktop.git
cd CivicSense-Desktop
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
python desktop.py
```

On Linux, install the Qt webview backend too:

```bash
python -m pip install 'pywebview[qt]'
```

The first launch downloads about **2.1 GB** of Qwen weights plus the embedding model. Open **Local models** to see preparation status or retry after an error. Later launches reuse the cache. Initial downloads need internet; cached models can run offline. No document content is sent to a model service.

If building `llama-cpp-python` fails, install its publisher-provided CPU wheel before retrying:

```bash
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Source builds require a C/C++ toolchain. Windows needs the WebView2 runtime; macOS uses WebKit; Linux needs a graphical session and Qt or GTK system libraries. See [desktop setup and troubleshooting](docs/DESKTOP.md). Source launch and executable build instructions are included; this repository does not claim verified installers for every OS.

A browser fallback uses the same local engine:

```bash
python desktop.py --browser
```

## Try it

1. Launch CivicSense and wait for **Local models** to report ready.
2. Upload [the fictional demonstration policy](examples/sample_policy.txt).
3. Ask “How much notice is required to cancel membership?”
4. Inspect the answer, explanation and source text. The policy says seven days.
5. Ask about parking: the supplied policy does not provide that information.

Only one document is indexed at a time. A successful upload replaces the previous index and clears the displayed conversation. Failed uploads preserve the current document. New workspace clears the index and conversation. Uploads are processed through temporary files and deleted after indexing; extracted text and chat remain in memory until the app closes. Chat history is a display of independent document questions, not conversational model memory. Scanned PDFs need OCR before upload. Maximum upload size is 25 MB.

## Research and paper

The original evaluation scripts, per-model outputs and reports are included. [Project map](docs/PROJECT_MAP.md) identifies the current app, historical experiments, optional legacy code and excluded local artifacts. [Research reproduction](docs/RESEARCH.md) explains how to obtain LegalBench-RAG and run the experiments.

The saved Phase 3 Qwen artifact reports:

| Measure | Saved value |
|---|---:|
| Queries | 500 |
| Valid answers | 495 |
| Operational error rate | 1.0% |
| BERTScore F1, valid answers | 0.8460 |
| Recall@5, document-path metric | 99.6% |
| MRR | 0.996 |
| Mean latency on the original setup | 18.3985 s |

Source: [raw Phase 3 results](model_results/result_v3_qwen_unrestricted.json). These are historical results, not measurements of this desktop release. The original model quantization is not established by the saved run metadata. The desktop changes inference engines and pins Q4_K_M, so exact quality and latency must be measured again.

**Read [the evidence audit](docs/EVIDENCE_AUDIT.md) before making conference slides.** The manuscript contains inconsistent retrieval and hardware claims. The saved security labels do not establish general prompt-injection resistance. BERTScore and JSON validity do not establish factual accuracy. Original artifacts remain unchanged so the research record is inspectable.

The [paper directory](paper/) contains the supplied manuscripts and planning notes. Referenced figure image files were not present in the workspace; the manuscript cannot compile as supplied without them. The [conference outline](docs/CONFERENCE_OUTLINE.md) provides an evidence-based starting point for the later presentation.

## Development

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python scripts/check_release.py
```

Tests use deterministic embeddings and fake model responses; they do not download model weights. CI runs these checks on Linux, Windows and macOS. [Build instructions](docs/DESKTOP.md) describe creating a native executable on each target OS.

The backend retains `/upload`, `/query`, `/clear_index` and `/load_file` for research. `/load_file` is restricted to `CIVICSENSE_EVALUATION_ROOT`, defaulting to `LegalBench-RAG/corpus`. The app binds to loopback and rejects foreign browser origins. It is a single-user desktop service, not a multi-user web deployment.

## Configuration

No `.env` is necessary. Optional settings are documented in [.env.example](.env.example). Set `CIVICSENSE_DATA_DIR` to change where models are cached. Default locations are:

| OS | Cache directory |
|---|---|
| Windows | `%LOCALAPPDATA%/CivicSense` |
| macOS | `~/Library/Application Support/CivicSense` |
| Linux | `$XDG_DATA_HOME/CivicSense`, or `~/.local/share/CivicSense` |

For the historical LM Studio experiments, use `CIVICSENSE_BACKEND=lmstudio`. The app follows the single available generation model, or uses an explicitly configured `LM_STUDIO_MODEL`; it never substitutes a hardcoded `local-model` ID.

## Publication and licenses

Secrets, service-account files, user uploads, Python environments, downloaded weights and third-party corpus files are excluded from Git. Get models and datasets from their publishers. Review [third-party notices](THIRD_PARTY_NOTICES.md), including the **Qwen Research License** and PyMuPDF licensing, before redistribution or commercial use. Making this repository public does not itself grant a software license; project authors must select one for their own code.
