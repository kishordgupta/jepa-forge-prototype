#!/usr/bin/env python3
"""Build evidence-derived Markdown, PDF, and compact publication artifacts."""
from __future__ import annotations
import gzip
import hashlib
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from build_expanded_report import DISPLAY, METHODS, test_counts
from validate_selection import read, validate_payload


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(data):
    rows = []
    for final in data["final_evaluations"]:
        groups = {}
        for run in final["runs"]:
            for method in run["evaluation"]:
                metric = "macro_f1" if "macro_f1" in method["test"] else "rmse"
                groups.setdefault((method["method"], metric), []).append(method["test"][metric])
        selection = next(s for s in data["selections"] if s["dataset"] == final["dataset"])
        for (method, metric), values in groups.items():
            rows.append({"dataset": final["dataset"], "task": final["task"]["name"], "method": method,
                         "metric": metric, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                         "n": len(values), "context_features": len(final["task"]["context"]),
                         "target_features": len(final["task"]["target"]),
                         "validation_mean_score": selection["lock"]["candidates"][selection["lock"]["selected_index"]]["mean_validation_score"]})
    return rows


def main():
    raw = (ROOT/"artifacts/selection/results.json").read_bytes()
    data = json.loads(raw)
    config = read(ROOT/"configs/selection.json")
    validation = read(ROOT/"results/selection_validation.json")
    validate_payload(data, config)
    assert validation["status"] == "passed" and validation["results_sha256"] == hashlib.sha256(raw).hexdigest()
    assert validation["datasets"] == 23 and validation["gpu_training_runs"] == 273
    tests = test_counts(ROOT/"results/selection_test_report.xml")
    rows = aggregate(data)
    look = {(r["dataset"], r["method"]): r for r in rows}
    names = config["datasets"]
    comparisons = {}
    for method in ("raw_context_linear", "raw_context_extra_trees", "random_encoder_linear"):
        counts = {"wins": 0, "ties": 0, "losses": 0}
        for name in names:
            jepa, other = look[name,"jepa_context_linear"], look[name,method]
            advantage = (jepa["mean"]-other["mean"])*(1 if jepa["metric"]=="macro_f1" else -1)
            counts["wins" if advantage>1e-12 else "losses" if advantage < -1e-12 else "ties"] += 1
        comparisons[method] = counts
    def val(row):
        return f"{row['mean']:.3f} +/- {row['std']:.3f}"
    intro = ("Automatic context/target selection is implemented and validated across 23 public/synthetic datasets. "
             "The study evaluates 91 schema-based candidates with three seeds each: 273 additional 60-epoch GPU training runs. "
             "All 23 selections were locked before test evaluation. Only selected tasks were evaluated on the test partition (69 seed evaluations). "
             "Together with the separate earlier 138-run study, this gives 411 full GPU training runs; smoke tests are excluded.")
    boundary = ("The selector receives copied training/validation rows only. It compares equal context budgets and identical forecasting targets. "
                "Training labels fit classification probes, and validation labels rank candidates; JEPA training remains label-free. "
                "Candidate choice uses mean validation macro F1 or negative RMSE across all seeds, with fixed-order ties. "
                "The test partition is used only after a durable global selection lock.")
    limitation = ("These are structure-based candidate templates, not learned causal feature importance or an exhaustive search. "
                  "Generic tabular blocks need domain review. Probe tuning and mask ranking share validation, so validation can be optimistic. "
                  "Data are the same 23 previously studied datasets with a new predeclared split (3026); this is not independent external replication. "
                  "One split, three seeds, bounded public samples, missing subject identifiers in some sources, and synthetic forecasting limit generalization.")
    quality = (f"The selected JEPA encoder beats raw linear on {comparisons['raw_context_linear']['wins']}/23 datasets, "
               f"Extra Trees on {comparisons['raw_context_extra_trees']['wins']}/23, and the random encoder on {comparisons['random_encoder_linear']['wins']}/23. "
               f"The low-rank heuristic triggers in {validation['low_rank_warning_runs']}/273 candidate runs. "
               "These descriptive comparisons do not establish statistical significance or a general JEPA advantage. "
               "Unselected candidates were not scored on test, so this study does not estimate test-set improvement over every possible feature division.")
    security = ("The privacy audit found no credentials under the applied scans. Two historical JUnit reports contained a personal machine name; "
                "their published replacements remove that metadata. The publication process replaces the affected branch ancestry with a sanitized snapshot. "
                "Gitleaks and metadata checks cover staged files, branch history, archives, and PDF text; a fake credential was correctly blocked. "
                "Local commit/push hooks and GitHub CI repeat the checks. GitHub's settings state branch protection is not enforced for this private repository's current account setup; "
                "native secret-scanning/push-protection controls were not offered in the observed settings. "
                "This cannot erase copies or platform-retained objects, or guarantee every future manual/API upload. See SECURITY_AUDIT.md for scope and final publication verification.")
    doc = ["# JEPA-FORGE automatic task-selection report", "", intro, "", "## What the extension does", "", boundary, "",
           "```mermaid", "flowchart LR", ' A["Dataset and schema"] --> B["Train / validation copies"]',
           ' B --> C["Equal-budget candidates"]', ' C --> D["GPU training + validation probes"]',
           ' D --> E["Lock all 23 choices"]', ' E --> F["Final selected-task test evaluation"]', "```", "",
           "## Final selected-task results", "", "Mean +/- sample standard deviation over three seeds. Macro F1 is higher-is-better; RMSE is lower-is-better. Each baseline receives the same context as its selected task.", "",
           "| Dataset | Chosen context | Features | Metric | JEPA + linear | Raw linear | Extra Trees |", "|---|---|---:|---|---:|---:|---:|"]
    for name in names:
        j = look[name,"jepa_context_linear"]
        doc.append(f"| {DISPLAY.get(name,name)} | {j['task']} | {j['context_features']} | {j['metric']} | {val(j)} | {val(look[name,'raw_context_linear'])} | {val(look[name,'raw_context_extra_trees'])} |")
    doc.extend(["", quality, "", "## Candidate rankings and exact feature choices", "",
                "Validation scores below determine selection. Negative RMSE is reported as positive RMSE for readability; lower is better. Classification uses higher macro F1."])
    for selection in data["selections"]:
        lock = selection["lock"]
        doc.extend(["", "### "+DISPLAY.get(selection["dataset"],selection["dataset"]), "",
                    "| Candidate | Validation mean | Sample SD | Selected |", "|---|---:|---:|---|"])
        for i,candidate in enumerate(lock["candidates"]):
            v = candidate["mean_validation_score"]*(1 if lock["metric"]=="macro_f1" else -1)
            doc.append(f"| {candidate['task']['name']} | {v:.6f} | {candidate['std_validation_score']:.6f} | {'yes' if i==lock['selected_index'] else ''} |")
        task = lock["selected_task"]
        doc.extend(["", f"Context columns (zero-based): `{task['context']}`.", "", f"Target columns (zero-based): `{task['target']}`."])
    doc.extend(["", "## Verification", "", f"{tests['passed']} local tests passed. Independent validation replayed all {validation['checkpoint_cpu_replays']} checkpoints on CPU training samples and all {validation['validation_probe_replays']} development probes. Maximum GPU-to-CPU embedding error was {validation['checkpoint_max_absolute_error']:.3g}, within the 1e-4 absolute/relative tolerance. All 23 selected exports reloaded successfully.", "",
                "GPU tensor, gradient and loss evidence identifies MPS execution without fallback. Data preparation, covariance checks and sklearn probes use CPU. Wall time includes monitoring and evaluation; it is not a hardware benchmark.", "",
                "## Privacy and publication", "", security, "", "## Limits", "", limitation, "",
                "## Reproduce", "", "```bash", "python scripts/run_selection.py --config configs/selection.json --output artifacts/selection", "python scripts/validate_selection.py", "python scripts/run_tests_private.py", "python scripts/build_selection_report.py", "python scripts/check_privacy.py --worktree", "```", "",
                "Use a fresh output directory for each rerun; existing selection locks are not overwritten. Public-data sources and licenses: docs/DATASETS.md and docs/PUBLIC_DATASETS_EXPANDED.md. Synthetic mechanisms: docs/SYNTHETIC_DATASETS_EXPANDED.md. Full protocol: docs/SELECTION_PROTOCOL.md.", ""])
    (ROOT/"SELECTION_REPORT.md").write_text("\n".join(doc))
    (ROOT/"results/selection_summary.json").write_text(json.dumps({"comparisons": comparisons, "rows": rows},indent=2)+"\n")
    compressed = ROOT/"results/selection_benchmark.json.gz"
    with compressed.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
            stream.write(raw)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=colors.HexColor("#103851"), spaceAfter=16))
    styles.add(ParagraphStyle(name="BodyCustom", fontSize=10, leading=14, spaceAfter=12))
    styles.add(ParagraphStyle(name="CellCustom", fontSize=7.7, leading=10))
    styles.add(ParagraphStyle(name="SmallCustom", fontSize=8.5, leading=11, spaceAfter=9))
    story = []
    def paragraph(text, style="BodyCustom"):
        return Paragraph(escape(text), styles[style])
    def table(content, widths):
        formatted = [[paragraph(str(cell),"CellCustom") for cell in row] for row in content]
        t = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dcecf4")),("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("LINEBELOW",(0,0),(-1,0),0.7,colors.HexColor("#577487")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f7f9")]),
                               ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        return t
    story += [paragraph("JEPA-FORGE\nAutomatic feature-division selection", "TitleCustom"), paragraph("Validated extension | 23 datasets | September 16, 2026", "SmallCustom"), paragraph(intro),
              paragraph("How a feature division is chosen", "Heading2"), paragraph(boundary)]
    story.append(table([["Step", "Decision and evidence"],["1. Propose", "Schema templates; equal observation budgets; fixed forecasting horizon."],["2. Compare", "Train three GPU seeds per candidate; score validation probes."],["3. Lock", "Save candidate rankings, feature indices, hashes and selected checkpoints for all datasets."],["4. Evaluate", "Only then evaluate the selected task on held-out test rows."]],[80,436]))
    story += [Spacer(1,14),paragraph(quality),PageBreak()]
    for classification in (True, False):
        title = "Selected classification tasks" if classification else "Selected forecasting tasks"
        story += [paragraph(title,"TitleCustom"),paragraph("Mean +/- sample SD over three seeds. " + ("Macro F1: higher is better." if classification else "RMSE in each dataset's own units: lower is better; do not compare magnitudes across datasets."),"SmallCustom")]
        content = [["Dataset / selected context", "Cols", "JEPA", "Raw linear", "Extra Trees"]]
        for name in names:
            j = look[name,"jepa_context_linear"]
            if (j["metric"]=="macro_f1") != classification: continue
            content.append([DISPLAY.get(name,name)+" / "+j["task"].replace("_"," "), j["context_features"], val(j), val(look[name,"raw_context_linear"]), val(look[name,"raw_context_extra_trees"])])
        story += [table(content,[192,30,98,98,98]),Spacer(1,12),paragraph("All baseline comparisons use the selected context. Full-information classification reference, PCA, random-encoder and persistence results are provided in the complete JSON/Markdown artifacts.","SmallCustom"),PageBreak()]
    story += [paragraph("Validation and limits","TitleCustom"),paragraph(f"{tests['passed']} local tests passed, including adversarial test-data poisoning, role/group separation, fixed-horizon budgets, checkpoint-lock checks, archive scanning, and redacted privacy findings."),
              paragraph(f"Independent verification: {validation['checkpoint_cpu_replays']} checkpoint CPU replays, {validation['validation_probe_replays']} probe replays and {validation['selected_export_reloads']} selected export reloads. Maximum embedding replay error: {validation['checkpoint_max_absolute_error']:.3g}."),
              paragraph(limitation),paragraph("Privacy controls and remaining boundary","Heading2"),paragraph(security),
              paragraph("Reproduction and full evidence","Heading2"),paragraph("Run scripts/run_selection.py, scripts/validate_selection.py and scripts/build_selection_report.py. See configs/selection.json and docs/SELECTION_PROTOCOL.md. The Markdown report lists every candidate score and the exact selected feature indices. Compressed JSON preserves complete histories, device evidence and rankings. The source repository remains private.","SmallCustom")]
    pdf = ROOT/"output/pdf/JEPA_FORGE_Automatic_Selection_Report.pdf"
    def footer(canvas, document):
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#526b7b"))
        canvas.drawString(42,24,"JEPA-FORGE | Automatic task-selection study")
        canvas.drawRightString(558,24,str(document.page))
    SimpleDocTemplate(str(pdf),pagesize=(600,792),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=42,
                      title="JEPA-FORGE automatic task-selection report",author="JEPA-FORGE prototype",subject="Public and synthetic dataset research prototype").build(story,onFirstPage=footer,onLaterPages=footer)
    receipt = {"result_uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
               "published_files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                   [compressed, ROOT/"SELECTION_REPORT.md",pdf,ROOT/"results/selection_summary.json",ROOT/"results/selection_validation.json",ROOT/"results/selection_test_report.xml"]}}
    (ROOT/"results/selection_publication.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"datasets":23,"candidates":91,"gpu_runs":273,"comparisons":comparisons,"local_tests":tests["passed"],"pdf":str(pdf.relative_to(ROOT)),"gzip_bytes":compressed.stat().st_size},indent=2))


if __name__ == "__main__":
    main()
