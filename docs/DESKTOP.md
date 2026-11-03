# Desktop setup

CivicSense is a Python desktop application using pywebview, a loopback Flask backend, SentenceTransformers and llama-cpp-python. The default model runs in the same process; no separate model server is needed. The app opens while model preparation continues in a background thread. The branded startup screen reports download and load states, then opens the workspace when the models are ready. Setup failures provide Retry setup and Open model settings controls. The Models screen keeps the same status available after startup.

Use Python 3.12, 64-bit. CPU inference is the portable default. The release does not automatically change to a different model on smaller hardware and does not silently use a cloud service. Insufficient memory must be resolved by freeing memory or using a more capable computer. A universally compatible executable for every desktop is not possible: architecture, OS webview and native inference dependencies still apply.

## Platform requirements

- Windows 10/11, x64: install Python 3.12 and Microsoft Edge WebView2 Runtime. For a source inference-engine build, install Visual Studio C++ Build Tools, or use the publisher's prebuilt CPU wheel.
- macOS: install Python 3.12 matching your machine's architecture. WebKit provides the webview. Source builds require Xcode command line tools. Apple Silicon needs an ARM64 Python environment.
- Linux: a graphical desktop, Python 3.12, and `pywebview[qt]` (or the GTK backend with its system dependencies). A C++ compiler and CMake may be required for llama-cpp-python.

See the publisher documentation: [pywebview installation](https://pywebview.flowrl.com/guide/installation.html) and [llama-cpp-python installation](https://llama-cpp-python.readthedocs.io/en/latest/).

On Linux/Windows, a CPU-only Torch install avoids downloading unnecessary CUDA packages:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-desktop.txt
```

## Text size

Use Ctrl + (or Ctrl =) to increase text size, Ctrl − to decrease it, and Ctrl 0 to reset. Cmd works too on macOS. Text sizes range from 70% to 200% for the current window, including the question field, answers, source passages and settings. The current percentage appears in the bottom status bar.

## Offline use

Launch online once and wait for Qwen and MiniLM to finish preparing. Keep their cache directory intact. Subsequent use reuses those model files. For a strict offline launch, set `HF_HUB_OFFLINE=1` in the environment; missing files then fail with a visible setup error instead of attempting a network lookup. Model weights are pinned to the Qwen repository revision in `model_runtime.py`. Embeddings use the upstream MiniLM repository and its cache.

An interrupted Hugging Face download can be resumed by retrying preparation. A missing inference dependency is detected before downloading the large generation model. The application never executes an OS installer or asks for administrator privileges.

## Build an executable and installers

Build on each target OS; PyInstaller is not a cross-compiler. Start in an activated environment with the desktop dependencies installed:

```bash
python -m pip install -r requirements-build.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python scripts/build_desktop.py
python scripts/package_desktop.py
```

On Linux also install `pywebview[qt]` before building. The output is a directory under `dist/CivicSense` (and a macOS app bundle on macOS). Distribute the entire output directory, not just its executable. Models are downloaded to the user's data directory on first launch and are not bundled in Git or the executable. Native packages are large because Torch and the inference engine are included.

Unsigned builds may trigger platform security prompts. Signing and notarization require publisher credentials and are outside the source build. Validate generated executables on clean target machines before calling them a supported release. The GitHub Actions workflow builds Windows x64, Linux x64 and macOS Apple Silicon packages. It checks frozen resources/native dependencies before packaging. Windows builds include a per-user Inno Setup installer and portable ZIP; macOS includes a DMG and APP ZIP; Linux includes a tar.gz executable archive. Tagged builds publish a preview release only when all jobs pass; manual runs leave artifacts in Actions. The app still downloads its model weights automatically. Intel macOS and Windows/Linux ARM packages are not provided.

## Troubleshooting

- **No module named llama_cpp:** install `requirements-desktop.txt` in the interpreter used to launch the app. If compilation fails, try the official CPU wheel index from the README.
- **No usable GUI backend:** install the platform webview dependency. `python desktop.py --browser` uses your browser with the same local pipeline.
- **Download failed:** check the connection, disk space and access to Hugging Face, then use Prepare / retry models. No API key is needed for the default public models.
- **No readable text:** scanned PDFs need OCR. Password-protected or damaged documents may also fail extraction.
- **No answer / invalid JSON:** the error is shown instead of silently presenting a malformed object. Try a narrower question or shorter document.
- **Context window exceeded:** the pipeline preserves the paper's five 350-word passages and 8192-token context. Token-heavy documents or long questions can exceed the context budget. The error remains visible; reduce question length.
- **Slow generation:** the default uses CPU, so performance can be slower than the original GPU-assisted experiments. The app does not claim the paper's 18.4-second latency for every desktop.

The app is for one local user and one active document. Do not expose its port to a network or run it as a shared server.

Packaged diagnostics: run the executable with `--self-test --self-test-report check.json`. Add `--with-models` to prepare the real models and run a document question. The default self-test uses no model downloads. The bundled `FIRST_RUN.txt`, sample policy and dependency license directory accompany native packages.

The publisher’s llama-cpp-python 0.3.36 macOS arm64 wheel failed ZIP integrity checks during release preparation. The macOS workflow builds the same version from source with Metal disabled for the portable CPU configuration. Source users on macOS should follow the explicit compiler/setup commands in the README. Windows and Linux use their validated wheels.

macOS packaging refreshes and verifies the app’s ad-hoc code signature after first-run notes and dependency licenses are added. This integrity check does not provide an Apple Developer ID or notarization.
