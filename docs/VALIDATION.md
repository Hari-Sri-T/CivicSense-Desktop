# Desktop validation

Local checks performed on 4 November 2026:

- 26 deterministic Python tests passed under Python 3.12, covering upload/query, source metadata, input validation, index replacement, JSON repair/schema validation, model lifecycle, external model ID resolution, corpus path boundaries and browser-origin restrictions.
- A headless Chromium check exercised the interface: document upload, a question, source inspection and toggling, model/about navigation, workspace reset, and 1360/900/720-pixel widths. No JavaScript page errors were observed. Model responses and embeddings were substituted for this interface check; it is not a generation-quality test.
- The native Qt/pywebview window loaded the application from the launcher's temporary loopback server and closed cleanly. Model preparation was bypassed only in this window-lifecycle check.
- A separate real-model check automatically downloaded Qwen2.5-3B-Instruct Q4_K_M, loaded the CPU inference engine, indexed the fictional sample policy, and returned valid answer/explanation objects for cancellation notice and parking questions. The responses correctly identified seven days' notice and the absence of parking arrangements. The initial run reused the checkout's local MiniLM weights. A second run from a clean Git archive, with no local settings or weights, automatically downloaded MiniLM, reused the pinned Qwen cache, and passed both questions again.
- `pip check`, Python compilation, JavaScript syntax validation, shell syntax validation and the release-file credential/size check passed.

The welcome-screen preview is a screenshot of the running interface. It does not show generated answers or represent a research measurement.

Windows/macOS native execution and packaged executables have not been locally verified. The GitHub workflows provide cross-platform unit checks and manually triggered candidate builds; their results must be inspected before publishing installers. Full benchmark replication, independent factuality scoring, OCR, conversational model memory, and signing/notarization were not part of this change.

The black/grey surfaces, blue model/control accents and keyboard text resizing were checked in Chromium. Ctrl +, Ctrl =, Ctrl −, Ctrl 0, keypad + and Cmd + changed the rendered font size as expected; the 70–200% bounds and preservation of typed input were verified. This optional browser test is `tests/test_appearance.py` and needs Playwright plus Chromium.

The branded startup screen was checked through loading, download messages, setup failure, retry, readiness and dismissal into model settings. The generated PNG loaded successfully. All 28 Python/browser checks passed after this change; the existing upload/query/source UI check also passed. `docs/loading-preview.png` captures a simulated loading state for visual review, not a runtime benchmark.

Desktop packaging verification: the Linux x64 PyInstaller bundle passed its offline native/resource diagnostic and a real-model question using cached Qwen/MiniLM. Its native window opened and loaded the packaged interface. The saved answer correctly stated seven days’ written notice. All 30 source/browser/packaging checks passed. Windows/macOS packages are built and checked by the release workflow; signing/notarization remains unconfigured.
