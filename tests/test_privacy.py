"""Credential-like fixtures are assembled at runtime, never committed as keys."""
import gzip
import importlib.util
import io
from pathlib import Path
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("privacy", Path(__file__).resolve().parents[1]/"scripts/check_privacy.py")
privacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(privacy)


@pytest.mark.parametrize("payload,rule", [
    (lambda: "/"+"Users"+"/"+"example_person"+"/source", "personal_home_path"),
    (lambda: "person"+"@"+"example.org", "personal_email"),
    (lambda: "https://overleaf.com/"+"project/"+"0123456789", "private_overleaf_link"),
    (lambda: 'hostname="'+"personal-device"+'.local"', "personal_machine_name"),
    (lambda: "-----BEGIN "+"PRIVATE KEY-----", "private_key_material"),
])
def test_metadata_values_are_found_but_never_echoed(payload, rule):
    value = payload()
    found, texts = privacy.inspect_bytes("report.txt", value.encode())
    assert rule in {item["rule"] for item in found}
    assert value not in repr(found)


def test_nested_archive_is_scanned_and_scan_failures_block():
    stream = io.BytesIO()
    payload = ("name"+"@"+"example.org").encode()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("notes.txt.gz", gzip.compress(payload))
    found, texts = privacy.inspect_bytes("bundle.zip", stream.getvalue())
    assert any(item["rule"] == "personal_email" for item in found)
    assert len(texts) == 1
    assert privacy.inspect_bytes("broken.zip", b"broken")[0][0]["rule"] == "extraction_failed"
    assert privacy.inspect_bytes("unknown.bin", b"\x00")[0][0]["rule"] == "unscannable_binary"


@pytest.mark.parametrize("name", [".env", ".env.production", "private.pem", "credentials.json", "abcdef123456:.env", "bundle.zip!.env"])
def test_sensitive_filenames_are_blocked_in_history_and_archives(name):
    assert privacy.forbidden_filename(name)


def test_missing_credential_scanner_is_not_a_success(monkeypatch):
    monkeypatch.setattr(privacy, "entries", lambda mode: iter([("safe.txt", b"Hello")]))
    report = privacy.scan(gitleaks=None)
    assert report["status"] == "failed"
    assert report["findings"] == [{"path": ".", "rule": "credential_scanner_required"}]
