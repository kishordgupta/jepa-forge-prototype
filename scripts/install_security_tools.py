#!/usr/bin/env python3
"""Install a checksum-pinned Gitleaks binary inside ignored artifacts/."""
import hashlib
import io
from pathlib import Path
import platform
import tarfile
from urllib.request import urlopen

VERSION = "8.30.1"
CHECKSUMS = {
    "darwin_arm64": "b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5",
    "linux_x64": "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
}


def main():
    architecture = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(platform.machine())
    system = platform.system().lower()
    key = f"{system}_{architecture}"
    if key not in CHECKSUMS:
        raise SystemExit("No reviewed binary checksum for this platform")
    name = f"gitleaks_{VERSION}_{key}.tar.gz"
    url = f"https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/{name}"
    with urlopen(url, timeout=60) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != CHECKSUMS[key]:
        raise ValueError("Downloaded Gitleaks checksum differs from reviewed release")
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        binary = archive.extractfile("gitleaks").read()
    output = Path(__file__).resolve().parents[1]/"artifacts/security-tools/gitleaks"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary); output.chmod(0o755)
    print(f"Verified Gitleaks {VERSION} installed in artifacts/security-tools")


if __name__ == "__main__":
    main()
