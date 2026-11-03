"""Build on the target operating system with desktop dependencies installed."""
import importlib.util
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
           "--windowed", "--name", "CivicSense", "--add-data", "templates:templates",
           "--add-data", "static:static"]
for package in ("webview", "llama_cpp", "sentence_transformers", "transformers", "sklearn", "pymupdf"):
    command.extend(["--collect-all", package])
if sys.platform.startswith("linux"):
    if importlib.util.find_spec("PyQt6") is None:
        raise SystemExit("Install pywebview[qt] before building the Linux desktop")
    command.extend(["--hidden-import", "webview.platforms.qt"])
command.append("desktop.py")
subprocess.run(command, cwd=root, check=True)
print("Built candidate desktop package under dist/. Test it on a clean target machine.")
