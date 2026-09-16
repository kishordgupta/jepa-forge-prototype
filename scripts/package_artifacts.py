#!/usr/bin/env python3
"""Bundle only prototype sources and public/synthetic experiment artifacts."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = []
    for folder in ("src/jepa_forge", "tests", "scripts", "configs", "docs", "results", "output/pdf", "artifacts/benchmark", "artifacts/expanded", "artifacts/selection", ".githooks"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append(path)
    for name in ("README.md", "REPORT.md", "EXPANDED_REPORT.md", "SELECTION_REPORT.md", "SECURITY.md", "SECURITY_AUDIT.md", ".gitleaks.toml", "LICENSE", "pyproject.toml", "requirements-lock.txt", ".gitignore", ".gitattributes", ".github/workflows/tests.yml"):
        path = ROOT / name
        if path.exists():
            files.append(path)
    output = ROOT / "artifacts/JEPA_FORGE_Reproducibility.zip"
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("BUNDLE_SHA256.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(p)).hexdigest() == sha for p,sha in manifest.items())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(digest + "  " + output.name + "\n")
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size, "sha256": digest, "files": len(manifest)}))

if __name__ == "__main__":
    main()
