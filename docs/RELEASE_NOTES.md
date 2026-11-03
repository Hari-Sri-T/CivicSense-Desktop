CivicSense Desktop preview release

Download the package matching your computer:

- Windows x64: `CivicSense-Windows-x64-Setup.exe`, or extract the portable ZIP and open `CivicSense.exe`.
- macOS Apple Silicon: open the `.dmg`, drag `CivicSense.app` to Applications, and open it. A ZIP of the app is also included.
- Linux x64: extract the `.tar.gz` and run `CivicSense` inside the extracted directory.

Python is bundled; a separate Python installation and LM Studio are not required. On first launch, keep internet available while CivicSense automatically downloads Qwen2.5-3B-Instruct Q4_K_M (about 2.1 GB) and MiniLM. Subsequent launches reuse the model cache. Models are not included in the download packages.

This release includes the black/grey interface with blue accents, generated Civic Sense logo, startup/loading screen, source inspection, and Ctrl + / Ctrl - text resizing.

These are unsigned preview packages. Native dependency/resource self-checks run before publication, but they do not replace interactive testing on your hardware. Windows needs WebView2. The Linux package is built on Ubuntu 22.04 and requires a compatible graphical desktop. The macOS package is for Apple Silicon, not Intel Macs. Consult the README for platform requirements, sample questions, research context and limitations.

Verify downloads using SHA256SUMS.txt. The full source and third-party notices are available in this repository. The desktop runtime is separate from the historical paper experiments; exact paper quality and latency have not been re-established for these binaries.
