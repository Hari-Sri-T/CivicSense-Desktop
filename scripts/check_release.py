"""Check the publishable Git file list without displaying credential values."""
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
).decode().split("\0")
patterns = [re.compile(p.encode()) for p in (
    r"AIza[0-9A-Za-z_-]{30,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"gh[pousr]_[A-Za-z0-9_]{20,}", r"github_pat_[A-Za-z0-9_]{30,}", r"hf_[A-Za-z0-9]{20,}",
)]
failures = []
for name in dict.fromkeys(files):
    if not name:
        continue
    path = root / name
    if not path.is_file():
        continue
    if name in {".env", "env file", "service-account.json"}:
        failures.append(f"Private file selected: {name}")
    if path.stat().st_size > 50 * 1024 * 1024:
        failures.append(f"File exceeds release size limit: {name}")
        continue
    if any(pattern.search(path.read_bytes()) for pattern in patterns):
        failures.append(f"Possible credential in: {name}")
if failures:
    raise SystemExit("\n".join(failures))
print("Release file check passed: no recognized credentials or oversized files.")
