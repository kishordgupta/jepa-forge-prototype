#!/usr/bin/env python3
"""Fail closed on private metadata, then scan extracted text with Gitleaks.

Scans the Git index by default (including tracked ignored files), --worktree for
tracked and untracked nonignored files, or every unique blob in --history.
Nested ZIP/gzip members and PDF text/metadata are inspected. Findings never
include matched values. No cloud scanning service receives file contents.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 64*1024*1024
PATTERNS = {
    "personal_home_path": rb"/(?:Users|home)/[A-Za-z0-9_.-]+(?:/|\b)",
    "personal_email": rb"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "private_overleaf_link": rb"overleaf\.com/(?:project|read)/[A-Za-z0-9]+",
    "personal_machine_name": rb'hostname\s*=\s*["\x27](?!redacted["\x27]|anonymous["\x27])[A-Za-z0-9_. -]+["\x27]',
    "local_device_name": rb"\b[A-Za-z0-9_-]+\.(?:local|lan)\b",
    "private_key_material": rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----",
    "embedded_url_credentials": rb"https?://[^\s/:]+:[^\s/@]+@",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def forbidden_filename(name):
    name = re.sub(r"^[0-9a-f]{12}:", "", name).rsplit("!", 1)[-1]
    path = Path(name)
    lower = path.name.lower()
    return ((lower == ".env" or lower.startswith(".env.")) and lower != ".env.example"
            or lower in {"credentials.json", "credentials", "id_rsa", "id_ed25519", ".netrc", ".npmrc"}
            or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".keystore"}
            or any(part.lower() in {".ssh", ".aws", ".gnupg"} for part in path.parts))


def inspect_bytes(name, data, *, depth=0):
    """Return redacted findings and extracted text records; errors are findings."""
    findings, texts = [], []
    if forbidden_filename(name):
        findings.append({"path": name, "rule": "sensitive_filename"})
    if len(data) > MAX_BYTES or depth > 5:
        return findings+[{"path": name, "rule": "scan_limit_exceeded"}], texts
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                total = 0
                for entry in archive.infolist():
                    if entry.is_dir():
                        continue
                    total += entry.file_size
                    if entry.file_size > MAX_BYTES or total > MAX_BYTES:
                        raise ValueError("Archive exceeds scan budget")
                    child, extracted = inspect_bytes(name+"!"+entry.filename, archive.read(entry), depth=depth+1)
                    findings.extend(child); texts.extend(extracted)
            return findings, texts
        if name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                payload = stream.read(MAX_BYTES+1)
            child, extracted = inspect_bytes(name[:-3], payload, depth=depth+1)
            return findings+child, extracted
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            pdf = PdfReader(io.BytesIO(data))
            data = ("\n".join(page.extract_text() or "" for page in pdf.pages)+"\n"+str(pdf.metadata)).encode()
        if name.endswith(".png"):
            # PNG metadata chunks are text, compressed text or XML; image pixels
            # are outside a text scanner's scope. PIL is a matplotlib dependency.
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                data = json.dumps(image.info, default=str).encode()
        # Unsupported binary formats are blocked instead of silently skipped.
        if b"\x00" in data:
            return findings+[{"path": name, "rule": "unscannable_binary"}], texts
        for rule, pattern in PATTERNS.items():
            for match in re.finditer(pattern, data):
                value = match.group()
                if rule == "personal_email" and value.endswith(b"@users.noreply.github.com"):
                    continue
                findings.append({"path": name, "rule": rule, "line": data[:match.start()].count(b"\n")+1})
        texts.append((name, data))
        return findings, texts
    except Exception as error:
        return findings+[{"path": name, "rule": "extraction_failed", "error_type": type(error).__name__}], texts


def entries(mode):
    if mode == "history":
        # Scan publishable branches, tags and remote-tracking refs. Tool-owned
        # recovery refs and reflogs remain local and are never pushed.
        for line in git("rev-list", "--objects", "--branches", "--tags", "--remotes").decode().splitlines():
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            sha, name = parts
            if git("cat-file", "-t", sha).strip() == b"blob":
                yield f"{sha[:12]}:{name}", git("cat-file", "blob", sha)
    elif mode == "worktree":
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
        for name in sorted(set(filter(None, names))):
            path = ROOT/name
            if path.is_symlink():
                yield name, b"\x00"  # Never follow a link outside the publication boundary.
            elif path.is_file():
                yield name, path.read_bytes()
    else:
        for item in git("ls-files", "--stage", "-z").decode().split("\0"):
            if not item:
                continue
            header, name = item.split("\t", 1)
            mode, sha, stage = header.split()
            yield name, git("cat-file", "blob", sha) if mode in {"100644", "100755"} and stage == "0" else b"\x00"


def scan(mode="index", gitleaks=None):
    findings, count, digests = [], 0, {}
    with tempfile.TemporaryDirectory(prefix="jepa-privacy-") as folder:
        temporary = Path(folder)
        payload = temporary/"payload"; payload.mkdir()
        paths = {}
        for name, data in entries(mode):
            count += 1
            digests[name] = hashlib.sha256(data).hexdigest()
            issues, records = inspect_bytes(name, data)
            findings.extend(issues)
            for source, body in records:
                leaf = f"{len(paths):06d}.txt"
                paths[leaf] = source
                (payload/leaf).write_bytes(body)
        leaks = []
        if gitleaks is not None:
            report = temporary/"gitleaks.json"
            command = [str(Path(gitleaks).resolve()), "dir", str(payload), "--redact=100", "--no-banner",
                       "--ignore-gitleaks-allow", "--gitleaks-ignore-path", str(temporary),
                       "--config", str(ROOT/".gitleaks.toml"), "--report-format=json", "--report-path", str(report)]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed.returncode not in (0, 1) or not report.exists():
                findings.append({"path": ".", "rule": "credential_scanner_failed"})
            else:
                for leak in json.loads(report.read_text()):
                    leaks.append({"path": paths.get(Path(leak["File"]).name, "unknown"),
                                  "rule": leak["RuleID"], "line": leak["StartLine"]})
            findings.extend(leaks)
        else:
            findings.append({"path": ".", "rule": "credential_scanner_required"})
    return {"status": "failed" if findings else "passed", "mode": mode,
            "files_or_blobs": count, "extracted_text_records": len(paths),
            "credential_findings": len(leaks), "findings": findings, "sha256": digests,
            "limits": "Pattern scanning is not a guarantee against all sensitive information; binary pixels are not OCR scanned."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--worktree", action="store_true")
    group.add_argument("--history", action="store_true")
    parser.add_argument("--gitleaks", default="artifacts/security-tools/gitleaks")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = scan("history" if args.history else "worktree" if args.worktree else "index", args.gitleaks)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k,v in report.items() if k != "sha256"}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
