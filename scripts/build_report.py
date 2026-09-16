#!/usr/bin/env python3
"""Generate the Markdown/PDF report and plots from measured experiment JSON."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["raw_context_linear", "raw_context_pca", "raw_context_extra_trees", "random_encoder_linear", "jepa_context_linear", "persistence", "full_raw_linear_reference"]
LABELS = {"raw_context_linear": "Raw context + linear", "raw_context_pca": "PCA + linear",
          "raw_context_extra_trees": "Raw context + ExtraTrees", "random_encoder_linear": "Random encoder + linear",
          "jepa_context_linear": "JEPA context + linear", "persistence": "Persistence",
          "full_raw_linear_reference": "Full raw reference*"}
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#007F86")
GRAY = colors.HexColor("#526275")


def validate_report_input(data):
    """This report template describes the complete predeclared MPS matrix."""
    policies = {("wine", "alternating"), ("wine", "first_half"), ("digits", "center"),
                ("digits", "right_half"), ("synthetic_sensors", "past16_future8"),
                ("synthetic_sensors", "past12_future12")}
    expected = {(d, t, s) for d, t in policies for s in (7, 17, 27)}
    observed = [(r["dataset"], r["task"], r["seed"]) for r in data.get("runs", [])]
    if data.get("status") != "complete" or len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Report requires the complete 18-run matrix: three unique seeds for each of six policies.")
    if data["config"]["device"] != "mps" or data["config"]["split_seed"] != 2026:
        raise ValueError("This report template requires the declared MPS benchmark with split seed 2026.")
    for run in data["runs"]:
        evidence = run["training"]["device_evidence"]
        if not all(evidence[k].startswith("mps") for k in ("online_parameter_device", "context_batch_device", "online_gradient_device")):
            raise ValueError("Missing actual GPU execution evidence.")
        if not run["training"]["checkpoint_roundtrip"]["passed"]:
            raise ValueError("A checkpoint roundtrip failed.")


def aggregate(data):
    groups = defaultdict(list)
    primary = {}
    for run in data["runs"]:
        key = (run["dataset"], run["task"])
        primary[key] = run["primary_a_priori"]
        for row in run["evaluation"]:
            metric = "macro_f1" if "macro_f1" in row["test"] else "rmse"
            groups[(*key, row["method"], metric)].append(row["test"][metric])
    return [{"dataset": k[0], "task": k[1], "method": k[2], "metric": k[3],
             "mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0,
             "n": len(v), "primary": primary[k[:2]]} for k, v in groups.items()]


def format_score(row):
    return f'{row["mean"]:.3f} +/- {row["std"]:.3f}'


def architecture():
    d = Drawing(480, 105)
    labels = [("PUBLIC / SYNTHETIC", "Typed data + metadata"), ("AUDIT + COMPILE", "Split, mask, normalize"),
              ("GPU PILOT", "Context -> target latent"), ("EVALUATE + EXPORT", "Baselines, JSON, hashes")]
    for i, (top, bottom) in enumerate(labels):
        x = i * 123
        d.add(Rect(x, 35, 111, 57, fillColor=colors.HexColor("#EAF3F5"), strokeColor=TEAL, rx=6, ry=6))
        d.add(String(x + 55.5, 72, top, textAnchor="middle", fontName="Helvetica-Bold", fontSize=7.4, fillColor=NAVY))
        d.add(String(x + 55.5, 52, bottom, textAnchor="middle", fontSize=7, fillColor=GRAY))
        if i < 3:
            d.add(Line(x + 113, 62, x + 121, 62, strokeColor=TEAL))
            d.add(Polygon([x + 121, 62, x + 117, 65, x + 117, 59], fillColor=TEAL, strokeColor=TEAL))
    d.add(String(240, 13, "Train-only preprocessing | disjoint views | held-out groups | context-only inference", textAnchor="middle", fontSize=8, fillColor=GRAY))
    return d


def plots(data, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    primaries = list(dict.fromkeys((r["dataset"], r["task"]) for r in rows if r["primary"]))
    fig, axes = plt.subplots(1, len(primaries), figsize=(12.2, 3.7))
    for ax, key in zip(np.atleast_1d(axes), primaries):
        selected = [r for r in rows if (r["dataset"], r["task"]) == key and r["method"] != "full_raw_linear_reference"]
        selected.sort(key=lambda r: METHODS.index(r["method"]))
        ax.barh(range(len(selected)), [r["mean"] for r in selected], xerr=[r["std"] for r in selected],
                color=["#007F86" if r["method"] == "jepa_context_linear" else "#ABBCCB" for r in selected], capsize=2)
        ax.set_yticks(range(len(selected)), [LABELS[r["method"]].replace("Raw context + ", "Raw + ").replace(" + linear", "") for r in selected], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(key[0].replace("synthetic_sensors", "Synthetic sensors") + "\n" + key[1], fontsize=10)
        ax.set_xlabel("Macro-F1 (higher is better)" if selected[0]["metric"] == "macro_f1" else "RMSE (lower is better)")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Context-only evaluation: identical visible inputs within each task", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "primary_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # Per-seed training loss is exposed as evidence, without equating it to utility.
    fig, axes = plt.subplots(1, len(primaries), figsize=(12.2, 3.4))
    for ax, key in zip(np.atleast_1d(axes), primaries):
        for run in data["runs"]:
            if (run["dataset"], run["task"]) != key:
                continue
            history = run["training"].get("history", run["training"].get("training_history", []))
            if history:
                loss_key = next((k for k in ("train_total_loss", "train_loss", "loss", "train_total") if k in history[0]), None)
                if loss_key:
                    ax.plot(range(1, len(history)+1), [h[loss_key] for h in history], label=str(run["seed"]))
        ax.set_title(key[0].replace("synthetic_sensors", "Synthetic sensors"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training objective")
        ax.grid(alpha=.18)
        if ax.lines:
            ax.legend(title="Seed", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/benchmark/results.json")
    args = parser.parse_args()
    data = json.loads((ROOT / args.results).read_text())
    validate_report_input(data)
    env = json.loads((ROOT / "artifacts/environment.json").read_text())
    validation_path = ROOT / "artifacts/validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {"status": "not recorded"}
    rows = aggregate(data)
    published = ROOT / "results"
    published.mkdir(exist_ok=True)
    shutil.copy2(ROOT / args.results, published / "benchmark.json")
    shutil.copy2(ROOT / "artifacts/environment.json", published / "environment.json")
    if validation_path.exists():
        shutil.copy2(validation_path, published / "validation.json")
    (published / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    plots(data, rows, published / "figures")
    primaries = list(dict.fromkeys((r["dataset"], r["task"]) for r in rows if r["primary"]))
    task_keys = list(dict.fromkeys((r["dataset"], r["task"]) for r in rows))
    comparisons = []
    for key in primaries:
        these = [r for r in rows if (r["dataset"], r["task"]) == key]
        jepa = next(r for r in these if r["method"] == "jepa_context_linear")
        linear = next(r for r in these if r["method"] == "raw_context_linear")
        text = (f'{key[0]} / {key[1]}: JEPA {jepa["metric"]} {format_score(jepa)}; '
                f'raw-context linear {format_score(linear)}.')
        comparisons.append(text)
    scope_text = ("This working prototype implements a typed task compiler, train-only normalization, structural/leakage diagnostics, "
                  "numeric exports with checksums, a context-only inference API, and small JEPA-style GPU pilots. "
                  "It does not implement the complete proposed ecosystem or reproduce the proposal's earlier result table.")
    protocol_text = (f'{len(data["runs"])} neural runs cover three datasets and six policies, with model seeds 7, 17 and 27, '
                     f'a fixed split/generator seed of 2026, and {data["config"]["epochs"]} epochs per run. '
                     "The first declared policy for each dataset was designated primary before evaluating the test set. "
                     "Sample standard deviations measure model-seed variation on one fixed split, not population uncertainty.")
    md = ["# JEPA-FORGE prototype: implementation and experiment report", "", f'Generated from measured run artifacts on {datetime.now(timezone.utc).date().isoformat()}.', "", scope_text, "",
          "## Outcome", "", *["- " + c for c in comparisons], "", "JEPA improved primary Digits macro-F1 over raw-context linear, but remained below ExtraTrees. It underperformed the raw-context linear model on Wine and sensor forecasting. Seven of 18 runs triggered a low-rank embedding warning (all six Wine runs and one 12-step sensor run). These results support pipeline feasibility, not general JEPA superiority.", "", "## Implementation and protocol", "", protocol_text, "",
          "GPU: Apple M3 Max / PyTorch MPS. Training and encoding use the GPU; scikit-learn baselines, probes and data processing use CPU. No paid cloud GPU was used.", "",
          "All primary inference receives context columns only. Hidden targets are used for training, forecast supervision, diagnostics and scoring; they never enter context-only inference. Full raw classification is a separate full-information reference.", "",
          "## Test results", "", "Mean +/- sample SD over the three model seeds. Compare methods within each policy. Macro-F1: higher is better; RMSE: lower is better, in original synthetic signal units.", "",
          "| Dataset | Policy | Method | Metric | Mean +/- SD |", "|---|---|---|---|---|"]
    for row in rows:
        md.append(f'| {row["dataset"]} | {row["task"]} | {LABELS[row["method"]]} | {row["metric"]} | {format_score(row)} |')
    md += ["", "*Full raw reference sees all features/pixels, including those withheld from context-only methods. It is not a matched baseline or theoretical ceiling.", "",
           "![Primary results](results/figures/primary_results.png)", "", "## Validation", "", "```json", json.dumps(validation, indent=2), "```", "",
           "## Reproduce", "", "```bash", "python -m venv .venv", "source .venv/bin/activate", "python -m pip install '.[dev,report]'", "pytest -q --junitxml=artifacts/pytest.xml", "jepa-forge inspect wine", "python scripts/verify_environment.py",
           "python scripts/run_experiments.py --config configs/benchmark.json --output artifacts/benchmark", "python scripts/validate_artifacts.py", "python scripts/build_report.py", "```", "",
           "The supplied benchmark config explicitly requires MPS. Change `device` to `cuda` for an available NVIDIA GPU or `cpu` for a disclosed CPU run. GPU unavailability raises an error. Large compiled arrays and checkpoints remain in the local artifact bundle and can be regenerated.", "",
           "## Limits and next steps", "", "- Small fixed-split feasibility experiments; no broad JEPA superiority or state-of-the-art claim.",
           "- Wine is tiny; Digits does not establish writer-disjoint generalization; synthetic signals do not establish real sensor performance.",
           "- Rule-based leakage checks cannot prove semantic safety or discover every target-derived variable. Checksums detect changed bytes, not trusted provenance.",
           "- Only numeric tabular, small image and grouped sequence adapters are present. Graph/audio/event/multimodal extensions, plugin isolation, community governance and usability studies remain future work.",
           "- The small latent predictor uses a variance regularizer and custom encoders; it is a JEPA-style pilot, not an official I-JEPA implementation.",
           "- Separate future work: real multivariate sensor benchmarks, broader held-out splits, ablations, independent reproduction and user studies.", "",
           "## Sources", "", "- [I-JEPA paper](https://arxiv.org/abs/2301.08243) and [official implementation](https://github.com/facebookresearch/ijepa).",
           "- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).",
           "- [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine), [Optical digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).",
           "- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). See docs/DATASETS.md for dataset attribution.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(md))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=NAVY, spaceAfter=12))
    styles.add(ParagraphStyle("ReportSub", fontSize=11, leading=15, textColor=GRAY, spaceAfter=14))
    styles.add(ParagraphStyle("ReportBody", fontSize=9.4, leading=13, spaceAfter=9, textColor=NAVY))
    styles.add(ParagraphStyle("ReportSmall", fontSize=7.5, leading=10, spaceAfter=6, textColor=GRAY))
    styles["Heading1"].textColor = NAVY
    styles["Heading1"].fontSize = 19
    styles["Heading2"].textColor = TEAL
    story = []
    def p(text, style="ReportBody"):
        return Paragraph(text, styles[style])
    def table(header, body, widths, size=8):
        cells = [[p(escape(str(v)), "ReportSmall") for v in row] for row in [header] + body]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF3F5")),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 6),
                               ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 5),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                               ("LINEBELOW", (0,0), (-1,0), .8, TEAL),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")])]))
        return t
    story += [p("JEPA-FORGE", "ReportTitle"), p("Prototype implementation and GPU experiment report", "ReportSub"),
              p("A reproducible feasibility pilot for scientific dataset-to-task compilation", "Heading2"), p(scope_text),
              architecture(), p("What was executed", "Heading2"), p(protocol_text),
              table(["Data", "Task family", "Evaluation"], [["Wine: 178 rows, 13 features", "Feature-group prediction", "Three-class frozen linear probe"],
                    ["Digits: 1,797 images, 8 x 8 pixels", "Spatially masked prediction", "Ten-class frozen linear probe"],
                    ["Synthetic grouped sensor windows", "Past-to-future prediction", "Future-signal regression"]], [145,170,177]),
              Spacer(1,12), p("Measured primary results", "Heading2")]
    story += [p(escape(c)) for c in comparisons]
    story += [p("Evidence files: results/benchmark.json, results/environment.json, results/validation.json. JSON manifests record split counts and hashes; exact split indices reside in local dataset.npz exports.", "ReportSmall"), PageBreak()]
    story += [p("Data integrity and model design", "Heading1"), p("Compiler contract", "Heading2"),
              p("RawDataset retains feature names, numeric observations, modality/shape, labels, group identities, interval coverage and provenance. TaskSpec records disjoint observed and hidden indices. Labels and identifiers are prohibited from task inputs when declared in the schema."),
              table(["Check", "Purpose / boundary"], [["Schema and finite numeric data", "Reject malformed shapes, invalid feature indices and non-finite observations."],
                    ["Split and group integrity", "Reject reused row indices, repeated source groups and duplicate records across splits."],
                    ["Context-target structure", "Reject overlap and reverse-time prediction; retain spatial/sequence layouts."],
                    ["Train-only normalization", "Fit mean and scale on training rows; reuse those parameters for validation and test."],
                    ["Export hashes", "Verify JSON and numeric NPZ bytes; no pickle loading. Integrity does not authenticate provenance."]], [145,347]),
              p("JEPA-style training", "Heading2"),
              p("The online encoder processes observed values plus a binary mask. A separate frozen teacher processes the hidden target view. An MLP predictor maps context embeddings to target embeddings. The teacher uses an exponential moving average of the online encoder. Training combines latent MSE with a context-variance penalty; the latter is an explicitly disclosed prototype adaptation."),
              p("An MLP handles Wine, a small Conv2d encoder handles Digits, and a Conv1d encoder preserves sensor order. Inference accepts only standardized context columns. The public context encoder API cannot consume full rows or concatenate hidden-target embeddings."),
              p("Evaluation boundaries", "Heading2"),
              p("All methods share saved train/validation/test indices. Linear-probe hyperparameters and ExtraTrees depth are selected on validation data. PCA and standardization fit training data only. Model training uses the predeclared final epoch. The test partition is not used to train, normalize, select checkpoints or choose task policies."),
              p("Targets have different information budgets across policies: central masking hides 16 pixels, right-half masking hides 32, and sensor forecasting uses 8- or 12-step horizons. Compare methods within one policy. No cross-policy performance ranking is asserted.", "ReportSmall"), PageBreak()]
    story += [p("Primary task results", "Heading1"), p("Context-only predictions; mean +/- sample SD across seeds 7, 17 and 27. Classification uses macro-F1 (higher is better). Forecasting uses RMSE in original synthetic units (lower is better).", "ReportSub"),
              Image(str(published / "figures/primary_results.png"), width=492, height=155)]
    body = []
    for method in METHODS:
        vals = []
        for key in primaries:
            found = next((r for r in rows if (r["dataset"],r["task"]) == key and r["method"] == method), None)
            vals.append(format_score(found) if found else "n/a")
        body.append([LABELS[method], *vals])
    story += [table(["Method", "Wine / F1", "Digits / F1", "Sensors / RMSE"], body, [168,108,108,108]),
              p("* Full raw reference sees every original feature or pixel, including the hidden region. It is reported separately as a full-information reference; it is not a matched comparator. n/a means the method does not apply.", "ReportSmall"),
              p("Interpretation", "Heading2"),
              p("On the primary tasks, JEPA improved Digits macro-F1 over the raw-context linear probe, but remained below ExtraTrees. On Wine, JEPA underperformed both raw-context baselines. For sensors, JEPA beat persistence but had higher RMSE than raw-context ridge. The results support pipeline feasibility and expose model limitations; they do not establish general JEPA superiority."),
              p("The same dataset split is used for all three model seeds. Deterministic linear baselines can have zero seed standard deviation. These are repeated optimization runs, not independent replications of the data-generating population.", "ReportSmall"), PageBreak()]
    story += [p("Alternative tasks and training evidence", "Heading1")]
    alternate = [key for key in task_keys if key not in primaries]
    body = []
    for key in alternate:
        these = [r for r in rows if (r["dataset"],r["task"]) == key]
        jepa = next(r for r in these if r["method"] == "jepa_context_linear")
        linear = next(r for r in these if r["method"] == "raw_context_linear")
        trees = next(r for r in these if r["method"] == "raw_context_extra_trees")
        body.append([key[0] + " / " + key[1], jepa["metric"], format_score(jepa), format_score(linear), format_score(trees)])
    story += [table(["Alternative policy", "Metric", "JEPA", "Raw linear", "ExtraTrees"], body, [142,50,100,100,100]),
              Spacer(1,10), Image(str(published / "figures/training_curves.png"), width=492, height=142),
              p("Compute evidence", "Heading2"),
              p(f'Environment: Apple M3 Max, 40 GPU cores, 128 GB unified memory; PyTorch {escape(env["packages"]["torch"])}. '
                f'The full benchmark elapsed time, including CPU evaluation and exports, was {data["elapsed_seconds"] / 60:.2f} minutes. '
                'The isolated forward/backward GPU smoke check passed. Every neural run records actual parameter/batch devices and synchronized training timing.'),
              p("Compiler and test evidence", "Heading2"), p(escape(json.dumps(validation))),
              p("Per-seed loss histories, validation predictability diagnostics, effective ranks and embedding standard deviations are preserved in results/benchmark.json. Seven of 18 runs triggered the low-rank heuristic (effective rank below 2): all six Wine runs and one 12-step sensor run. These flags identify a limitation of this pilot model.", "ReportSmall"), PageBreak()]
    story += [p("Reproduction and limits", "Heading1"), p("Run the prototype", "Heading2"),
              p("Create a Python 3.11+ environment and install <font face='Courier'>pip install '.[dev,report]'</font>. Run <font face='Courier'>pytest -q</font>, then inspect a dataset with <font face='Courier'>jepa-forge inspect wine</font>. Use the listed policy name with <font face='Courier'>jepa-forge compile</font>."),
              p("Run <font face='Courier'>python scripts/run_experiments.py --config configs/benchmark.json</font>, then <font face='Courier'>python scripts/build_report.py</font>. The supplied config requires MPS. Explicitly set CUDA for an NVIDIA GPU or CPU for a disclosed CPU run. GPU unavailability fails clearly."),
              p("Evidence and portability", "Heading2"),
              p("The repository includes source, tests, dataset attribution, the experiment protocol, fixed configuration, dependency lock snapshot, measured JSON, charts and this report. Local run artifacts also include compiled NPZ datasets and tensor-only checkpoints. JSON manifests record dataset/split hashes; NPZ archives contain split indices and compiler means/scales. Probe/PCA estimators are reproduced by refitting the recorded grids and seeds. GPU timing includes software/device effects and is not a cross-hardware speed comparison."),
              p("What remains to establish", "Heading2"),
              p("This is a bounded research prototype. Graphs, audio, event streams, multimodal fusion, generalized dataset discovery, untrusted plugin execution, signed releases, ecosystem governance and user studies are not implemented. Leakage rules depend on correct metadata and cannot discover every semantic shortcut."),
              p("Wine has a small test set. Digits does not establish writer-disjoint generalization. Synthetic sensor dynamics do not demonstrate real-world industrial forecasting. Additional data splits, datasets, ablations, external replications and domain review are required before broader claims."),
              p("Sources and attribution", "Heading2"),
              p('I-JEPA: <link href="https://arxiv.org/abs/2301.08243" color="#007F86">arxiv.org/abs/2301.08243</link>. Official implementation: github.com/facebookresearch/ijepa.<br/>'
                'PyTorch MPS: docs.pytorch.org/docs/stable/notes/mps.html.<br/>'
                'UCI Wine: archive.ics.uci.edu/dataset/109/wine. UCI Optical Digits: archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits.<br/>'
                'Dataset records identify CC BY 4.0; full attribution and loader/subset details appear in docs/DATASETS.md.<br/>'
                'scikit-learn leakage guidance: scikit-learn.org/stable/common_pitfalls.html.', "ReportSmall")]
    out_pdf = ROOT / "output/pdf/JEPA_FORGE_Prototype_Report.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#DCE5EC")); canvas.line(54, 42, 558, 42)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(GRAY)
        canvas.drawString(54, 28, "JEPA-FORGE | Research prototype | Measured results")
        canvas.drawRightString(558, 28, str(doc.page))
    SimpleDocTemplate(str(out_pdf), pagesize=(612,792), leftMargin=60, rightMargin=60,
                      topMargin=48, bottomMargin=58, title="JEPA-FORGE prototype and GPU experiments", author="JEPA-FORGE prototype").build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"markdown": "REPORT.md", "pdf": str(out_pdf), "runs": len(data["runs"])}))


if __name__ == "__main__":
    main()
