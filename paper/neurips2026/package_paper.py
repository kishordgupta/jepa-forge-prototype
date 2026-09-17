"""Package the reviewed paper using explicit source/evidence allowlists."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAMP = (2026, 9, 17, 0, 0, 0)

SOURCE_README = """# Anonymous workshop paper source

JEPA-FORGE: Auditable Context-Target Selection for Predictive Representations

This is an author-review draft, not a submission or acceptance notice.
Import this ZIP into Overleaf. Set the main document to main.tex and compiler
to XeLaTeX. Alternatively run `latexmk -xelatex main.tex` or `tectonic main.tex`.
Figures and tables are already supplied; no Python or GPU is needed to compile.

The source has explicit Anonymous Authors and empty author PDF metadata.
The workshop style comes from the official NeurIPS 2026 kit; two public email
identifiers were removed from comments only. Executable style lines are unchanged.
The `nonanonymous` option displays our explicit anonymous author line rather than
unfilled affiliation placeholders. Preserve anonymity for double-blind review.

Before submission, human authors must review all claims, citations, AI-use
disclosure, authorship, and current workshop requirements. Remove the explicitly
marked draft-footer override only when preparing an actual submission.

The separate anonymous evidence archive includes code, measured results,
data attribution, and regeneration instructions. No checkpoint collection or
private repository history is included here.
"""

ARTIFACT_README = """# Anonymous JEPA-FORGE experiment artifact

This archive accompanies an author-review workshop draft. It is not evidence
of submission or acceptance. The historical study contains 273 MPS runs on
23 datasets. The new extension has 27 tasks on 25 underlying datasets, three
repeated splits, and 1,044 neural training trajectories. A separate transfer
study evaluates official frozen I-JEPA/V-JEPA models on two image subsets and
one synthetic-video task, plus 81 scratch training trajectories. Official
pretraining compute is not matched to the small scratch models.

## Verify the frozen evidence without GPU retraining

Use Python 3.11 or later from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,report]'
PYTHONPATH=src python paper/neurips2026/build_evidence.py
PYTHONPATH=src python paper/neurips2026/build_extension_evidence.py
```

The builder verifies the frozen result hash against the saved validation report,
checks payload consistency, and derives all manuscript figures and tables.
This check does not replay absent checkpoints. Reports include the historical
results/selection_validation.json and the new extension/transfer validation
reports. The original 169-test report and current expanded test report are
separately labeled. Saved reports are recorded evidence, not
a claim that those checks have just been rerun on another machine.

## Rerun the full study

The supplied configuration requires an Apple MPS GPU with fallback disabled.
It fails explicitly if the required accelerator is unavailable. To use CUDA,
change the device explicitly and label the new run; do not equate it with the
frozen MPS measurement. Public datasets are downloaded into a local cache and
the synthetic cases are generated from the supplied code. Source provenance,
licenses, grouping qualifications, and transformations are documented in docs/.

Install the experiments extra for new comparisons and use fresh output directories:

```sh
python -m pip install '.[experiments]'
PYTHONPATH=src python scripts/run_extension.py \\
  --output artifacts/new_extension
PYTHONPATH=src python scripts/validate_extension.py \\
  --directory artifacts/new_extension
PYTHONPATH=src python scripts/run_vision_transfer.py \\
  --output artifacts/new_vision
PYTHONPATH=src python scripts/validate_extension.py \\
  --directory artifacts/new_vision --vision
PYTHONPATH=src python scripts/run_tests_private.py \\
  --output artifacts/reproduction_tests.xml
```

These commands preserve the supplied frozen files in results/. The validation
script uses checkpoints and arrays produced by the new run. Exact historical
checkpoint bytes are not included and must not be claimed to have been verified
from this lightweight archive. The environment lock records the measured
development environment, while pyproject.toml specifies dependency ranges.
Some public-data tests require caches produced by the downloads and otherwise skip.
There is no promise of bitwise reproduction on different software or hardware.

## Scope and privacy

Included: source, tests, configurations, development security scripts, measured
compressed selection results, saved validation/test reports, manuscript source,
and result-derived tables/figures. Excluded: Git history, identities, credentials,
private proposal material, downloaded raw archives, and large model checkpoints.
No funding, institutional ethics approval, or public release is implied.

MANIFEST.json lists SHA-256 hashes for every other file in this archive. These
hashes establish internal byte integrity, not authenticity of the data sources.
The Apache-2.0 code license and synthetic/public data qualifications are supplied in
LICENSE and docs/. AI assistance is disclosed in the paper; human authors remain
responsible for the scientific content and any eventual submission.
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_zip(path, entries):
    manifest = {name: {"sha256": digest(data), "bytes": len(data)}
                for name, data in sorted(entries.items())}
    entries = dict(entries)
    entries["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return {"files": len(entries), "bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    args = parser.parse_args()
    out = ROOT / "output"
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    source_paths = [HERE / p for p in ("main.tex", "references.bib", "neurips_2026.sty")]
    source_paths += sorted((HERE / "figures").glob("*.pdf"))
    source_paths += [HERE / "generated" / f"{p}.tex" for p in
                     ("classification", "forecasting", "inventory", "other_baselines")]
    source_paths += sorted((HERE / "generated").glob("extension_*.tex"))
    source = {str(p.relative_to(HERE)): p.read_bytes() for p in source_paths}
    source["README.md"] = SOURCE_README.encode()
    source_report = write_zip(out / "JEPA_FORGE_NeurIPS2026_Overleaf.zip", source)

    evidence_paths = []
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"),
                            ("scripts", "*.py"), ("configs", "*.json"), ("docs", "*.md")):
        evidence_paths.extend(sorted((ROOT / folder).rglob(pattern)))
    evidence_paths += [ROOT / p for p in ("pyproject.toml", "requirements-lock.txt", "requirements-extension-lock.txt", "LICENSE", ".gitleaks.toml",
        ".gitignore", "results/selection_benchmark.json.gz", "results/selection_validation.json",
        "results/selection_test_report.xml", "results/extension_benchmark.json.gz",
        "results/extension_benchmark_validation.json", "results/vision_transfer.json.gz",
        "results/vision_transfer_validation.json", "results/extension_test_report.xml", "EXTENSION_REPORT.md")]
    evidence_paths += source_paths
    evidence_paths += [HERE / "build_evidence.py", HERE / "build_extension_evidence.py",
                       HERE / "generated/evidence.json", HERE / "generated/extension_evidence.json",
                       HERE / "generated/contexts.tex", HERE / "generated/candidates.tex",
                       HERE / "CLAIMS_AND_AUTHOR_REVIEW.md"]
    evidence = {str(p.relative_to(ROOT)): p.read_bytes() for p in evidence_paths}
    evidence["README.md"] = ARTIFACT_README.encode()
    evidence_report = write_zip(out / "JEPA_FORGE_NeurIPS2026_Anonymous_Artifact.zip", evidence)
    final_pdf = out / "pdf/JEPA_FORGE_NeurIPS2026_Workshop_Paper.pdf"
    shutil.copyfile(args.pdf, final_pdf)
    report = {"status": "author_review_draft_not_submitted", "overleaf": source_report,
              "anonymous_artifact": evidence_report,
              "pdf": {"bytes": final_pdf.stat().st_size, "sha256": digest(final_pdf.read_bytes())}}
    (out / "paper_package_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
