import hashlib
import json

import pytest

import desktop_check
from scripts.package_desktop import write_checksums


def test_archive_checksums(tmp_path):
    (tmp_path / "CivicSense.zip").write_bytes(b"desktop archive")
    (tmp_path / "CivicSense.tar.gz").write_bytes(b"linux archive")
    (tmp_path / "readme.txt").write_text("Not an archive")
    write_checksums(tmp_path)
    entries = (tmp_path / "SHA256SUMS.txt").read_text().splitlines()
    assert len(entries) == 2
    for line in entries:
        digest, name = line.split("  ")
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


def test_diagnostic_report_success_and_failure(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    monkeypatch.setattr(desktop_check, "check_desktop", lambda with_models: {"status": "passed", "models_checked": with_models})
    desktop_check.write_desktop_check(report, True)
    assert json.loads(report.read_text())["models_checked"] is True
    def fail(with_models):
        raise RuntimeError("Missing native library")
    monkeypatch.setattr(desktop_check, "check_desktop", fail)
    with pytest.raises(RuntimeError, match="Missing native library"):
        desktop_check.write_desktop_check(report)
    assert json.loads(report.read_text()) == {"status": "failed", "error": "Missing native library"}
