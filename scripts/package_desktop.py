"""Archive native builds and create Windows/macOS installers on their target OS."""
import argparse
import hashlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def write_checksums(directory):
    archives = sorted(path for path in directory.iterdir() if path.suffix in {".zip", ".gz", ".dmg", ".exe"})
    lines = []
    for path in archives:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_notices(destination):
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        for file in distribution.files or []:
            if file.name.lower().startswith(("license", "copying", "notice")):
                source = distribution.locate_file(file)
                if source.is_file():
                    target = destination / f"{name}-{distribution.version}" / file.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0-preview.3")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+-]*", args.version):
        parser.error("Invalid release version")
    root = Path(__file__).resolve().parents[1]
    output = root / "dist/archives"
    output.mkdir(parents=True, exist_ok=True)
    architecture = {"x86_64": "x64", "AMD64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
    if architecture is None:
        raise SystemExit(f"Unsupported architecture: {platform.machine()}")
    bundle = root / ("dist/CivicSense.app" if sys.platform == "darwin" else "dist/CivicSense")
    if not bundle.is_dir():
        raise SystemExit("Build CivicSense first")
    resources = bundle / "Contents/Resources" if sys.platform == "darwin" else bundle
    shutil.copy2(root / "packaging/FIRST_RUN.txt", resources / "FIRST_RUN.txt")
    shutil.copy2(root / "THIRD_PARTY_NOTICES.md", resources / "THIRD_PARTY_NOTICES.md")
    shutil.copy2(root / "examples/sample_policy.txt", resources / "sample_policy.txt")
    collect_notices(resources / "DEPENDENCY-LICENSES")
    if sys.platform == "win32":
        shutil.make_archive(str(output / f"CivicSense-Windows-{architecture}"), "zip", bundle.parent, bundle.name)
        compiler = shutil.which("ISCC.exe") or str(Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Inno Setup 6/ISCC.exe")
        subprocess.run([compiler, f"/DAppVersion={args.version}", f"/DSourceDir={bundle}",
                        f"/DOutputDir={output}", str(root / "packaging/windows.iss")], check=True)
    elif sys.platform == "darwin":
        # Adding first-run notes and licenses changes the bundle resource seal.
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(bundle)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(bundle)], check=True)
        name = f"CivicSense-macOS-{architecture}"
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(output / f"{name}.zip")], check=True)
        with tempfile.TemporaryDirectory(prefix="civicsense-dmg-", dir=root / "dist") as directory:
            staging = Path(directory)
            shutil.copytree(bundle, staging / bundle.name, symlinks=True)
            (staging / "Applications").symlink_to("/Applications")
            shutil.copy2(root / "packaging/FIRST_RUN.txt", staging / "FIRST_RUN.txt")
            subprocess.run(["hdiutil", "create", "-volname", "CivicSense", "-srcfolder", str(staging),
                            "-format", "UDZO", "-ov", str(output / f"{name}.dmg")], check=True)
    elif sys.platform.startswith("linux"):
        with tarfile.open(output / f"CivicSense-Linux-{architecture}.tar.gz", "w:gz") as archive:
            archive.add(bundle, arcname=bundle.name)
    else:
        raise SystemExit(f"Unsupported platform: {sys.platform}")
    write_checksums(output)
    print(f"Native packages and checksums: {output}")


if __name__ == "__main__":
    main()
