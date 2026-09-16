#!/usr/bin/env python3
"""Run pytest and strip personal machine metadata before publishing its XML."""
import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def sanitize(path):
    tree = ET.parse(path)
    for node in tree.iter():
        for field in ("hostname", "file"):
            node.attrib.pop(field, None)
    tree.write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/selection_test_report.xml")
    args, extra = p.parse_known_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={args.output}", *extra])
    if Path(args.output).exists():
        sanitize(args.output)
    raise SystemExit(result.returncode)
