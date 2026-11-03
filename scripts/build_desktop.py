"""Build on the target OS; model weights are downloaded on first launch."""
import importlib.util
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
               "--windowed", "--name", "CivicSense", "--add-data", "templates:templates",
               "--add-data", "static:static", "--add-data", "examples:examples",
               "--add-data", "THIRD_PARTY_NOTICES.md:."]
    for package in ("webview", "llama_cpp", "sentence_transformers", "transformers", "sklearn", "pymupdf"):
        command.extend(["--collect-all", package])
    for package in ("pytest", "playwright", "PyQt5", "PySide2", "PySide6", "tkinter"):
        command.extend(["--exclude-module", package])
    if sys.platform.startswith("linux"):
        if importlib.util.find_spec("PyQt6") is None:
            raise SystemExit("Install pywebview[qt] before building the Linux desktop")
        command.extend(["--hidden-import", "webview.platforms.qt"])
    elif sys.platform == "darwin":
        command.extend(["--osx-bundle-identifier", "org.civicsense.desktop", "--icon", "static/civicsense-logo.png"])
    elif sys.platform == "win32":
        command.extend(["--icon", "static/civicsense-logo.png"])
    command.append("desktop.py")
    subprocess.run(command, cwd=root, check=True)
    print("Built desktop package under dist/. Run its --self-test before distributing.")


if __name__ == "__main__":
    main()
