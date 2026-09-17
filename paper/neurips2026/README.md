# JEPA-FORGE workshop manuscript

**JEPA-FORGE: Auditable Context-Target Selection for Predictive Representations**

Anonymous author-review draft prepared September 17, 2026. It has not been submitted or accepted. The proposed venue is the forthcoming non-archival track of PhysWorldAI at NeurIPS 2026 Atlanta. Consult `VENUE_RECOMMENDATION.md` for the posted dates, moderate scope fit, and unresolved portal/schedule discrepancies. Human authors must verify the manuscript and the operative workshop rules before submission.

## Files

- `main.tex`, `references.bib`, `neurips_2026.sty`: editable manuscript and official-format style.
- `figures/`: vector PDF figures generated from the frozen experiment.
- `generated/`: result-derived tables, exact contexts, candidate rankings, and `evidence.json`.
- `build_evidence.py`: validates and regenerates the historical study's evidence.
- `build_extension_evidence.py`: validates the new extension/transfer payloads and creates the new tables, figure, numeric macros and report.
- `CLAIMS_AND_AUTHOR_REVIEW.md`: evidence mapping, interpretation boundaries, and author decisions.
- `OPENREVIEW_STATUS.json`: public venue metadata checked during the deadline assessment; excludes contact details.
- `package_paper.py`: explicit file allowlist for the source and anonymous evidence archives.

The revised manuscript reports new GPU experiments: 27 tasks on 25 underlying datasets across three repeated splits, with 1,044 neural training trajectories, plus a separate official-model transfer study with 81 scratch trajectories. It adds raw MHEALTH/PAMAP2 activity and forecasting tasks, independent baseline selection, supervised encoders, TabM/CatBoost, feature-selection controls, and matched-budget fixed-mask trajectories. Official I-JEPA/V-JEPA backbones remain frozen and their external pretraining compute is not matched to the pilot. The original 273-run automatic-selection study remains separately labeled; the earlier 138 fixed-policy runs are excluded from these comparisons.

## Build and edit

From the repository root, with the project dependencies installed:

```sh
PYTHONPATH=src python paper/neurips2026/build_evidence.py
PYTHONPATH=src python paper/neurips2026/build_extension_evidence.py
tectonic --keep-logs --outdir build paper/neurips2026/main.tex
```

Create the output directory first if needed. Alternatively, use `latexmk -xelatex main.tex` from this directory. The distributed source archive has `main.tex` at its root; import it into Overleaf, set the main document to `main.tex`, and choose XeLaTeX. Building the paper from supplied tables and figures requires no GPU or dataset downloads. Regenerating those tables/figures requires the separate anonymous evidence archive or the full project tree.

The compiled draft has an explicit **Not submitted** footer. When preparing an actual submission after human review, remove only the draft notice override in `main.tex` and check the workshop's current format instructions. Keep `Anonymous Authors` for double-blind review. The `nonanonymous` style option displays that explicit anonymous author line instead of the template's unfilled affiliation/address placeholders; it does not reveal any author identity. Add names and affiliations only when the applicable track requests them.

## Style provenance

The style originates from the [official NeurIPS 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip), archive SHA-256:

`82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`

Two public contact identifiers were removed from comment-only style lines so that the package satisfies the project's strict no-email scan. All executable style lines are unchanged; original template attribution is retained.

## Artifact scope

The anonymous evidence archive includes the measured compressed results, the saved validation/test reports, current source, configurations, tests, and data documentation. It excludes Git history, credentials, proposal material, author-linked repository addresses, caches, raw downloaded data, and the large checkpoint/array collection. Public downloads and synthetic generators reconstruct the data. Regenerated results must be labeled separately from the supplied frozen study.

AI assistance covered implementation, experiment orchestration, analysis, and writing, as disclosed in the draft. It does not substitute for human authorship, scientific review, or responsibility.
