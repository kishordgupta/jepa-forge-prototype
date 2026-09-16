#!/usr/bin/env python3
"""Build the separate, evidence-derived 23-dataset expansion report.

Requires the complete 120-run expansion and its independent validation record.
The original 18-run benchmark remains a historical measurement snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_expanded import ValidationError, DEVICE_FIELDS, finite_tree, read_json, require, sha256, summarize, unique, validate_sources

DISPLAY = {
    "wine": "Wine", "digits": "Digits", "synthetic_sensors": "Original sensors",
    "breast_cancer": "Breast cancer (WDBC)", "ionosphere": "Ionosphere", "sonar": "Sonar",
    "semeion": "Semeion", "letter": "Letter", "pendigits": "PenDigits", "satimage": "SatImage",
    "har": "Human activity (HAR)", "dry_bean": "Dry Bean", "isolet": "ISOLET",
    "synthetic_lorenz": "Lorenz trajectories", "synthetic_mackey_glass": "Mackey-Glass delay",
    "synthetic_switching_var": "Switching VAR", "synthetic_chirp_seasonal": "Chirp + seasonal",
    "synthetic_coupled_oscillators": "Coupled oscillators", "synthetic_nonlinear_ar": "Nonlinear AR",
    "synthetic_nonlinear_multiview": "Nonlinear multiview", "synthetic_hierarchical_multiclass": "Hierarchical classes",
    "synthetic_sparse_interactions": "Sparse interactions", "synthetic_manifold_nuisance": "Manifold + nuisance",
}
METHODS = {
    "raw_context_linear": "Raw linear", "raw_context_pca": "PCA + linear",
    "raw_context_extra_trees": "Extra Trees", "random_encoder_linear": "Random encoder",
    "jepa_context_linear": "JEPA + linear", "full_raw_linear_reference": "Full-input reference",
    "persistence": "Persistence",
}


def label(name):
    return DISPLAY.get(name, name.replace("_", " ").title())


def score(row):
    return f"{row['mean']:.3f} +/- {row['std']:.3f}"


def test_counts(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    require(cases and failed == 0, "Report requires a nonempty passing JUnit report")
    return {"total": len(cases), "passed": len(cases)-skipped, "skipped": skipped, "failed": failed}


def check_matrix(data, datasets, runs):
    require(data.get("status") == "complete", "Report cannot use a partial or failed experiment")
    finite_tree(data)
    require(data["config"]["device"] == "mps", "Report requires MPS experiments")
    names, seeds = data["config"]["datasets"], data["config"]["model_seeds"]
    unique(names, "report datasets")
    unique(seeds, "report seeds")
    require(len(names) == datasets and seeds == [7, 17, 27], "Dataset/seed scope does not match the report")
    audits = [(item["dataset"], item["task"]) for item in data["task_audits"]]
    unique(audits, "report task audits")
    require(set(name for name, task in audits) == set(names), "Task dataset set differs from configuration")
    require(all(count == 2 for count in Counter(name for name, task in audits).values()), "Two task policies are required per dataset")
    actual = [(run["dataset"], run["task"], run["seed"]) for run in data["runs"]]
    unique(actual, "report runs")
    require(len(actual) == runs and set(actual) == {(name, task, seed) for name, task in audits for seed in seeds}, "Incomplete report run matrix")
    for run in data["runs"]:
        evidence = run["training"]["device_evidence"]
        require(all(evidence[field] == "mps:0" for field in DEVICE_FIELDS), "Actual GPU tensor evidence is missing")
        require(evidence["gpu_fallback_enabled"] is False, "CPU fallback cannot support the GPU claim")
        require(run["training"]["checkpoint_roundtrip"]["passed"] is True, "Checkpoint round-trip is missing")


def inputs(args):
    original, validation = read_json(args.original), read_json(args.validation)
    expanded_bytes = result_bytes(args.expanded)
    expanded = json.loads(expanded_bytes)
    check_matrix(original, 3, 18)
    check_matrix(expanded, 20, 120)
    require(set(original["config"]["datasets"]).isdisjoint(expanded["config"]["datasets"]), "Additional datasets duplicate the original three")
    require(validation.get("status") == "passed" and validation.get("completed_gpu_runs") == 120, "Independent expanded validation has not passed")
    require(validation.get("results_sha256") == hashlib.sha256(expanded_bytes).hexdigest(), "Validation is stale for the expanded result payload")
    require(validation.get("checkpoint_cpu_replay_checks") == 120, "Independent CPU checkpoint replay is incomplete")
    require((ROOT / "results/figures/expanded_comparison.png").is_file(), "Run scripts/plot_expanded.py before building the report")
    try:
        validate_sources(ROOT, ROOT / "configs/expanded.json", expanded["source_sha256"])
    except ValidationError:
        validate_sources(ROOT, ROOT / "configs/expanded.json", expanded["source_sha256"], archive=ROOT / "results/expanded_runtime.zip")
    return original, expanded, validation, test_counts(args.tests)


def result_bytes(path):
    """Read original JSON bytes from either local JSON or published gzip."""
    path = Path(path)
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def publish_evidence(source, destination):
    """Preserve exact measured JSON bytes in a deterministic gzip container."""
    raw = result_bytes(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            compressed.write(raw)
    require(gzip.decompress(destination.read_bytes()) == raw, "Published gzip does not round-trip")
    return {"path": str(destination), "compressed_bytes": destination.stat().st_size,
            "compressed_sha256": sha256(destination), "uncompressed_bytes": len(raw),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(), "gzip_mtime": 0}


def lookup(summary):
    return {(row["dataset"], row["task"], row["method"]): row for row in summary}


def compare(primary, rows, baseline):
    result = {"wins": 0, "ties": 0, "losses": 0}
    for task in primary:
        name, policy = task["dataset"], task["task"]
        jepa, other = rows[(name, policy, "jepa_context_linear")], rows[(name, policy, baseline)]
        advantage = jepa["mean"] - other["mean"]
        if jepa["metric"] == "rmse":
            advantage = -advantage
        result["wins" if advantage > 1e-12 else "losses" if advantage < -1e-12 else "ties"] += 1
    return result


def analyses(original, expanded):
    original_summary, expanded_summary = summarize(original), summarize(expanded)
    rows = lookup(expanded_summary)
    primary = [task for task in expanded["task_audits"] if task["primary_a_priori"]]
    require(len(primary) == 20 and len({task["dataset"] for task in primary}) == 20, "Exactly one primary policy per dataset is required")
    comparisons = {method: compare(primary, rows, method) for method in
                   ("raw_context_linear", "raw_context_extra_trees", "random_encoder_linear")}
    low = lambda data: sum(bool(run["training"]["final_train_context_diagnostics"]["heuristic_collapse_flag"]) for run in data["runs"])
    return {"original_summary": original_summary, "summary": expanded_summary, "rows": rows,
            "primary": primary, "comparisons": comparisons, "expanded_low_rank": low(expanded),
            "combined_low_rank": low(original)+low(expanded)}


def catalog_rows(primary):
    rows = []
    for audit in primary:
        metadata = audit["manifest"]["dataset"]["metadata"]
        dimensions = audit["report"]["dimensions"]
        group_counts = audit["report"].get("group_counts", {})
        rows.append({"name": audit["dataset"], "metadata": metadata, "dimensions": dimensions,
                     "splits": audit["split_sizes"], "groups": group_counts,
                     "primary": audit["task"], "raw_sha256": audit["manifest"]["dataset"]["raw_X_sha256"]})
    return rows


def build_markdown(original, expanded, validation, tests, analysis):
    comparisons, rows = analysis["comparisons"], analysis["rows"]
    total_datasets = len(original["config"]["datasets"])+len(expanded["config"]["datasets"])
    total_runs = len(original["runs"])+len(expanded["runs"])
    lines = ["# JEPA-FORGE: expanded 23-dataset prototype report", "",
             f"**{total_datasets} datasets and {total_runs} completed GPU training runs:** the original three-dataset, 18-run study plus **20 additional datasets and 120 new GPU runs**. The expansion contains 10 public datasets and 10 distinct synthetic generators, each with two declared task policies and three model seeds.", "",
             "These are fresh prototype measurements. Public data are bounded samples with new internal partitions; synthetic mechanisms test controlled behavior. This is not an official benchmark reproduction, a scientific-realism validation, or evidence of general JEPA superiority.", "",
             "## Evidence and observed comparisons", "",
             f"Independent validation confirms {validation['audited_tasks']} exported tasks, {validation['checkpoint_hash_checks']} checkpoint hashes, and {validation['checkpoint_cpu_replay_checks']} CPU checkpoint replays. Training tensors, parameters, losses, and gradients were recorded on `mps:0`. The full test report has **{tests['passed']} passed** and {tests['skipped']} skipped tests.", "",
             "| Primary-task comparison (20 datasets) | JEPA wins | Ties | Losses |",
             "| --- | ---: | ---: | ---: |"]
    for method, counts in comparisons.items():
        lines.append(f"| JEPA vs {METHODS[method]} | {counts['wins']} | {counts['ties']} | {counts['losses']} |")
    lines += ["", "Wins compare seed-mean test metrics within the same predeclared primary policy: higher macro F1 for classification, lower RMSE for forecasting. Counts weight datasets equally and are descriptive; they are not significance tests or model-selection decisions.", "",
              "![JEPA versus the matched raw-context linear baseline across the 20 new primary tasks](results/figures/expanded_comparison.png)", "",
              "The figure shows macro-F1 percentage-point changes for classification and relative RMSE reduction for forecasting. Positive values favor JEPA. Whiskers show one model-seed standard deviation, not confidence intervals. Forecast percentages remain within-system comparisons and are not averaged across generators.", "",
              f"The representation heuristic flagged **{analysis['expanded_low_rank']} of 120 new runs** and {analysis['combined_low_rank']} of 138 combined runs for low standard deviation or effective rank. A flag is a diagnostic, not an independent finding of complete collapse; its absence does not prove a useful representation.", "",
              "## Fixed protocol", "",
              f"- Model seeds: {expanded['config']['model_seeds']}; split/generator seed: {expanded['config']['split_seed']}; {expanded['config']['epochs']} fixed epochs per neural run; final checkpoint retained.",
              "- First declared policy is primary before test evaluation. Both policies are retained. Comparisons stay within a policy because masks, available context, and forecast horizons differ.",
              "- Training-only normalization and PCA. Encoders train without class labels. Frozen linear probes use validation macro F1 or RMSE to select the declared grid and remain training-fitted; test data do not select checkpoints, probes, or policies.",
              "- Logistic C / ridge alpha: 0.01, 0.1, 1, 10, 100. PCA: min(8, input dimensions, training rows minus 1). Extra Trees: 100 trees and depth 8 or unlimited.",
              "- JEPA-style latent prediction with an EMA target encoder and variance regularization; latent dimension 32, batch 128, Adam learning rate 0.001, EMA 0.99, variance weight 1.0. This is a small pilot adaptation, not a reproduction of a named published JEPA architecture.",
              "- PyTorch model training and encoding use the Apple GPU. Scikit-learn preprocessing/probes/baselines and covariance diagnostics use CPU. Validator checkpoint replay also uses CPU, with atol=rtol=1e-4.",
              "- Values below are mean +/- sample standard deviation over three model seeds on one fixed split. Deterministic baselines may have zero standard deviation; seeds are not independent dataset replications.", "",
              "## All 20 new primary tasks", "",
              "Macro F1 is a fraction and higher is better. RMSE is in each dataset's original synthetic signal units and lower is better; RMSE values are not comparable across different generators.", "",
              "| Dataset | Policy | Metric | JEPA | Raw linear | PCA | Extra Trees | Random encoder | Reference |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for task in analysis["primary"]:
        name, policy = task["dataset"], task["task"]
        get = lambda method: rows[(name, policy, method)]
        metric = get("jepa_context_linear")["metric"]
        special = "full_raw_linear_reference" if metric == "macro_f1" else "persistence"
        values = [score(get(method)) for method in ["jepa_context_linear", "raw_context_linear", "raw_context_pca", "raw_context_extra_trees", "random_encoder_linear", special]]
        lines.append(f"| {label(name)} | `{policy}` | {metric} | " + " | ".join(values) + " |")
    lines += ["", "The classification reference uses all input features and is not a matched context-only baseline or theoretical ceiling. The forecasting reference is persistence using the last observed value of each channel.", "",
              "## Dataset catalog, provenance, and limits", "",
              "Detailed dataset documentation: [public datasets](docs/PUBLIC_DATASETS_EXPANDED.md) and [synthetic generators](docs/SYNTHETIC_DATASETS_EXPANDED.md).", ""]
    for entry in catalog_rows(analysis["primary"]):
        name, meta, shape = entry["name"], entry["metadata"], entry["dimensions"]
        split = entry["splits"]
        lines += [f"### {label(name)} (`{name}`)", "",
                  f"{shape['rows']:,} retained rows, {shape['features']} flattened features; input shape `{meta.get('input_shape')}`. Training/validation/test rows: {split['train']:,}/{split['val']:,}/{split['test']:,}. Primary task: `{entry['primary']}` ({shape['context']} context / {shape['target']} target coordinates).",
                  f"Split policy: {meta.get('split_policy', 'See manifest')}." ]
        if entry["groups"]:
            lines.append("Groups by partition: " + ", ".join(f"{key}={value}" for key, value in entry["groups"].items()) + ".")
        if "source_full_rows" in meta:
            lines.append(f"Source size: {meta['source_full_rows']:,} rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `{meta['source_archive_sha256']}`.")
        if meta.get("source_url"):
            lines.append(f"[Source or mechanism reference]({meta['source_url']}).")
        if meta.get("citation"):
            lines.append("Attribution: " + meta["citation"])
        lines.append("Recorded data license: " + meta.get("license", "See source metadata") + ".")
        if meta.get("generator"):
            lines += ["Generator: " + meta["generator"], "",
                      "```json", json.dumps(meta["generator_parameters"], indent=2), "```"]
        for field in ("complexity_rationale", "scientific_claim_boundary", "scientific_limitation", "evaluation_limitation"):
            if meta.get(field):
                lines.append(meta[field])
        if meta.get("limitations"):
            lines += ["", "Recorded dataset limitations:", ""]
            limitations = meta["limitations"]
            for limitation in ([limitations] if isinstance(limitations, str) else limitations):
                lines.append("- " + str(limitation))
        lines += [f"Realized feature-array SHA-256: `{entry['raw_sha256']}`.", ""]
    lines += ["## Every baseline under every new policy", "",
              "This appendix retains all 40 policy tasks and all six methods. Full per-seed accuracy, macro F1, MAE/RMSE, validation histories, and diagnostics are preserved in the result JSON.", ""]
    for audit in expanded["task_audits"]:
        name, policy = audit["dataset"], audit["task"]
        lines += [f"### {label(name)} / `{policy}` ({'primary' if audit['primary_a_priori'] else 'secondary'})", "",
                  "| Method | Test metric | Mean +/- SD | Seeds | Information |",
                  "| --- | --- | ---: | ---: | --- |"]
        for row in analysis["summary"]:
            if row["dataset"] == name and row["task"] == policy:
                information = "Full input reference" if row["method"] == "full_raw_linear_reference" else "Context only"
                lines.append(f"| {METHODS[row['method']]} | {row['metric']} | {score(row)} | {row['n']} | {information} |")
        lines.append("")
    lines += ["## Original three-dataset snapshot", "",
              "The original 18 measurements are preserved rather than represented as reruns under the expanded source tree. Their runtime source files are archived in `results/original_runtime.zip`. Its historic hashes are audited separately; current runtime hash checks apply to the new 120-run expansion.", "",
              "| Dataset / primary policy | Metric | JEPA | Raw linear | Extra Trees |",
              "| --- | --- | ---: | ---: | ---: |"]
    old_rows = lookup(analysis["original_summary"])
    for audit in original["task_audits"]:
        if audit["primary_a_priori"]:
            key = (audit["dataset"], audit["task"])
            values = [old_rows[(*key, method)] for method in ("jepa_context_linear", "raw_context_linear", "raw_context_extra_trees")]
            lines.append(f"| {label(key[0])} / `{key[1]}` | {values[0]['metric']} | " + " | ".join(score(row) for row in values) + " |")
    lines += ["", "## Interpretation boundaries and next research work", "",
              "This expansion broadens structural coverage and exposes where a small encoder is competitive or limited. Dataset labels, sampling caps, synthetic equations, and one fixed split constrain the interpretation. Results do not establish clinical utility, physical calibration, generalization to unseen populations, real-world deployment quality, developer productivity, or broad state-of-the-art performance.", "",
              "Priority follow-up studies are independent splits or external cohorts; targeted analysis of low-rank runs and generator-specific failures; and stronger modality-specific architectures under matched compute and tuning budgets. Do not choose among the reported task policies using their test scores and then report that choice as independently validated.", "",
              "## Reproduction and evidence", "", "```bash",
              "python -m pip install '.[dev,report]'", "python -m pytest -q --junitxml=artifacts/expanded_pytest.xml",
              "python scripts/run_experiments.py --config configs/expanded.json --output artifacts/expanded",
              "python scripts/validate_expanded.py --config configs/expanded.json --results artifacts/expanded/results.json --output results/expanded_validation.json",
              "python scripts/plot_expanded.py",
              "python scripts/build_expanded_report.py", "```", "",
              "To rebuild the report from the published measured JSON without rerunning training or downloading datasets:", "",
              "```bash", "python scripts/build_expanded_report.py --expanded results/expanded_benchmark.json.gz", "```", "",
              "The gzip file preserves the original JSON bytes exactly and uses a deterministic timestamp of zero. Report validation compares the SHA-256 of its decompressed payload with the independent validation record. The compressed-file SHA is recorded separately in `results/expanded_publication.json`. Full binary export/checkpoint validation requires the local experiment artifacts or their reproducibility bundle.", "",
              "Use a process with access to the Metal GPU; the benchmark rejects silent CPU fallback. The validator requires 20 datasets, two policies each, three seeds, complete artifacts, and current runtime hashes. It performs no GPU training.", "",
              f"Expanded result SHA-256: `{validation['results_sha256']}`.",
              f"Expanded config SHA-256: `{validation['config_sha256']}`.",
              "The exported manifests retain precise context/target indices, split identities, normalization, source attribution, and generator parameters. Checksums detect modifications; they do not authenticate authorship.", ""]
    return "\n".join(lines)


def build_pdf(path, original, expanded, validation, tests, analysis):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate, Image, Paragraph, Spacer, Table, TableStyle, PageBreak

    navy, teal, gray = colors.HexColor("#16324A"), colors.HexColor("#087F8C"), colors.HexColor("#51606E")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=navy, spaceAfter=16))
    styles.add(ParagraphStyle(name="SectionX", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=navy, spaceBefore=12, spaceAfter=9, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyX", fontName="Helvetica", fontSize=10, leading=14, textColor=navy, spaceAfter=9))
    styles.add(ParagraphStyle(name="SmallX", fontName="Helvetica", fontSize=8.4, leading=11, textColor=gray, spaceAfter=7))
    styles.add(ParagraphStyle(name="CellX", fontName="Helvetica", fontSize=7.7, leading=10, textColor=navy))

    def p(text, style="BodyX"):
        text = str(text).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    def table(header, body, widths):
        result = Table([[p(value, "CellX") for value in header]] + [[p(value, "CellX") for value in row] for row in body], colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5F2F4")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, teal),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return result

    rows, comparisons = analysis["rows"], analysis["comparisons"]
    story = [p("JEPA-FORGE", "TitleX"), p("Expanded prototype: 23 datasets", "SectionX"),
             p("138 completed GPU runs\n120 new runs across 20 additional datasets", "SectionX"),
             p("The original three-dataset, 18-run study is preserved as a historical snapshot. The expansion adds 10 public datasets and 10 synthetic mechanisms, with two task policies and three fixed model seeds for every dataset."),
             p("This report measures a small JEPA-style task-compilation and representation-learning pilot. It does not claim official benchmark reproduction or general superiority."),
             p("Observed primary-task comparisons", "SectionX"),
             table(["JEPA compared with", "Wins", "Ties", "Losses"],
                   [[METHODS[method], value["wins"], value["ties"], value["losses"]] for method, value in comparisons.items()], [270, 78, 78, 78]),
             Spacer(1, 10), p("Counts compare the seed-mean test metric within each of 20 predeclared primary tasks. Classification uses higher macro F1; forecasting uses lower RMSE. Counts are descriptive and are not significance tests.", "SmallX"),
             p(f"Low-rank/low-variance heuristic: {analysis['expanded_low_rank']} of 120 new runs flagged; {analysis['combined_low_rank']} of 138 combined runs. A flag requires investigation and is not proof that every representation is constant.", "SmallX"),
             p(f"Evidence: {validation['audited_tasks']} export reloads, 120 checkpoint hashes and CPU replays; {tests['passed']} tests passed. Neural device evidence is mps:0. Classical estimators and validation replay use CPU.", "SmallX"), NextPageTemplate("landscape"), PageBreak()]

    story += [p("Results across all 20 new primary tasks", "SectionX"),
              Image(str(ROOT / "results/figures/expanded_comparison.png"), width=708, height=708*1332/2430),
              p("Positive values favor JEPA relative to the matched raw-context linear baseline. Classification changes are macro-F1 percentage points; forecast reductions are percentages of each system's raw-linear RMSE. Whiskers show one model-seed SD, not confidence intervals. Cross-generator RMSE values and percentage effects are not pooled. Extra Trees and other comparators remain in the report tables.", "SmallX"),
              NextPageTemplate("portrait"), PageBreak()]

    story += [p("Protocol and verification", "SectionX")]
    details = [
        f"Seeds 7, 17, 27; split/generator seed 2026; {expanded['config']['epochs']} fixed epochs; the final epoch is retained. First declared policy is primary before any test evaluation.",
        "Training-only feature scaling and PCA; separate labels for probes. Validation macro F1 or RMSE selects probe hyperparameters; the selected estimator remains fitted on training rows. Test observations do not select models, policies, or hyperparameters.",
        "Context-only comparisons: raw linear model, PCA plus linear probe, Extra Trees, random encoder plus linear probe, and trained JEPA encoder plus linear probe. Full-input classification references are separately labeled; forecasting also uses persistence.",
        "Logistic C and ridge alpha use 0.01, 0.1, 1, 10, 100. Extra Trees uses 100 trees and depth 8 or unlimited. PCA uses at most eight components. The JEPA pilot uses latent dimension 32, batch size 128, Adam 0.001, EMA 0.99, and variance weight 1.0.",
        "Reported dispersion is sample standard deviation across three model seeds on one fixed dataset split. Deterministic methods can have zero seed variation. These are optimization repeats, not independent data replications.",
        "GPU evidence includes parameters, batches, loss, and online gradients on mps:0, with a frozen EMA teacher and no silent CPU fallback. Export reloads validate schemas, splits and checksums. Independent checkpoint replay uses the first 32 training rows on CPU with absolute and relative tolerance 1e-4.",
        "Current source/configuration hashes are validated for the expansion. The original runtime is archived separately; its old measurements are not misrepresented as reruns under today's code.",
    ]
    story += [p(text) for text in details]
    story += [p("Scope limits", "SectionX"), p("Public datasets are bounded samples with new internal partitions; they are not original competition or UCI benchmark splits. Known entities are held out where identifiers are available. Synthetic equations are controlled stress tests, not calibrated scientific simulators. No clinical-utility, physical-validity, deployment, human-study, or broad state-of-the-art claim follows from this pilot."), PageBreak()]

    catalog = catalog_rows(analysis["primary"])
    for synthetic, heading in [(False, "Public dataset catalog"), (True, "Synthetic mechanism catalog")]:
        entries = [entry for entry in catalog if entry["name"].startswith("synthetic_") == synthetic]
        story.append(p(heading, "SectionX"))
        body = []
        for entry in entries:
            meta, dim = entry["metadata"], entry["dimensions"]
            rationale = meta.get("generator", meta.get("complexity_rationale", "See exported metadata"))
            groups = sum(entry["groups"].values()) if entry["groups"] else "rows"
            body.append([label(entry["name"]), f"{dim['rows']:,} x {dim['features']}", groups, rationale])
        story.append(table(["Dataset / mechanism", "Rows x features", "Groups", "Structural challenge"], body, [105, 76, 43, 280]))
        story += [Spacer(1, 9), p("Exact source URLs, attribution, generator parameters, split sizes, raw-array hashes, and all policy definitions appear in the Markdown report and exported manifests. Public dataset metadata record CC BY 4.0; synthetic outputs record CC0-1.0. Software licensing is separate.", "SmallX"), PageBreak()]

    for metric, heading in [("macro_f1", "New primary classification tasks"), ("rmse", "New primary forecasting tasks")]:
        body = []
        for audit in analysis["primary"]:
            key = (audit["dataset"], audit["task"])
            jepa = rows[(*key, "jepa_context_linear")]
            if jepa["metric"] != metric:
                continue
            body.append([label(key[0])] + [score(rows[(*key, method)]) for method in
                         ("jepa_context_linear", "raw_context_linear", "raw_context_extra_trees", "random_encoder_linear")])
        story += [p(heading, "SectionX"), p("Mean +/- sample SD, three seeds. " + ("Macro F1 is a fraction; higher is better." if metric == "macro_f1" else "RMSE is in original generator units; lower is better. Do not compare RMSE across different generators."), "SmallX"),
                  table(["Dataset", "JEPA", "Raw linear", "Extra Trees", "Random encoder"], body, [112, 98, 98, 98, 98]),
                  Spacer(1, 10), p("The primary policy was fixed in advance. Both policies and all six methods are retained in the complete Markdown appendix. Full-input classification references have more information and are not matched baselines; forecast persistence uses only the last observation.", "SmallX"), PageBreak()]

    story += [p("All 40 new policies: key comparisons", "SectionX"),
              p("Mean +/- sample SD across three seeds. F1 denotes macro F1 (higher is better); RMSE is lower-is-better. The Markdown report retains PCA, random-encoder, and full-input/persistence rows as well.", "SmallX")]
    body = []
    for audit in expanded["task_audits"]:
        key = (audit["dataset"], audit["task"])
        jepa = rows[(*key, "jepa_context_linear")]
        body.append([label(key[0]), key[1]+(" *" if audit["primary_a_priori"] else ""), "F1" if jepa["metric"] == "macro_f1" else "RMSE",
                     score(jepa), score(rows[(*key, "raw_context_linear")]), score(rows[(*key, "raw_context_extra_trees")])])
    story += [table(["Dataset", "Policy (* primary)", "Metric", "JEPA", "Raw linear", "Extra Trees"], body, [96, 99, 36, 91, 91, 91]), PageBreak()]

    story += [p("Original snapshot and provenance", "SectionX")]
    old_rows = lookup(analysis["original_summary"])
    body = []
    for audit in original["task_audits"]:
        if audit["primary_a_priori"]:
            key = (audit["dataset"], audit["task"])
            body.append([label(key[0]), key[1]] + [score(old_rows[(*key, method)]) for method in
                         ("jepa_context_linear", "raw_context_linear", "raw_context_extra_trees")])
    story += [table(["Dataset", "Original primary policy", "JEPA", "Raw linear", "Extra Trees"], body, [88, 125, 97, 97, 97]),
              Spacer(1, 9), p("Wine and Digits use macro F1; original sensors use RMSE. The 18 original runs are preserved with their historical runtime snapshot in results/original_runtime.zip. No current-source-equivalence claim is made for those older measurements.", "SmallX"),
              p("Public sources and attribution", "SectionX")]
    for entry in catalog:
        meta = entry["metadata"]
        if not entry["name"].startswith("synthetic_"):
            text = f"{label(entry['name'])}: {meta.get('citation', '')}\n{meta.get('source_url', '')}"
            story.append(p(text, "SmallX"))
    story += [p("Synthetic-source boundary", "SectionX"), p("Synthetic datasets are generated locally with recorded equations, seeds, and parameters. Named dynamics such as Lorenz or Mackey-Glass identify the implemented mechanism; they do not establish calibrated realism or exact reproduction of another benchmark. Mechanism reference links and full parameter records are in EXPANDED_REPORT.md.", "SmallX"),
              p("Evidence files", "SectionX"), p("results/expanded_validation.json; results/expanded_summary.json; artifacts/expanded/results.json; results/expanded_test_report.xml; EXPANDED_REPORT.md.", "SmallX"),
              p("Expanded result SHA-256: " + validation["results_sha256"], "SmallX")]

    def footer(canvas, doc):
        page_width = canvas._pagesize[0]
        canvas.setStrokeColor(colors.HexColor("#D9E1E7"))
        canvas.line(54, 40, page_width-54, 40)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(gray)
        canvas.drawString(54, 27, "JEPA-FORGE | 23 datasets | measured prototype evidence")
        canvas.drawRightString(page_width-54, 27, str(doc.page))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(str(path), pagesize=(612, 792), leftMargin=54, rightMargin=54,
                               topMargin=48, bottomMargin=54, title="JEPA-FORGE Expanded 23-Dataset Report", author="JEPA-FORGE prototype")
    document.addPageTemplates([
        PageTemplate(id="portrait", frames=[Frame(54, 54, 504, 690, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], pagesize=(612, 792), onPage=footer),
        PageTemplate(id="landscape", frames=[Frame(36, 54, 720, 510, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], pagesize=(792, 612), onPage=footer),
    ])
    document.build(story)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="results/benchmark.json")
    parser.add_argument("--expanded", default="artifacts/expanded/results.json")
    parser.add_argument("--validation", default="results/expanded_validation.json")
    parser.add_argument("--tests", default="results/expanded_test_report.xml")
    parser.add_argument("--markdown", default="EXPANDED_REPORT.md")
    parser.add_argument("--pdf", default="output/pdf/JEPA_FORGE_Expanded_23_Dataset_Report.pdf")
    parser.add_argument("--evidence", default="results/expanded_benchmark.json.gz")
    args = parser.parse_args(argv)
    original, expanded, validation, tests = inputs(args)
    publication = publish_evidence(args.expanded, args.evidence)
    analysis = analyses(original, expanded)
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(build_markdown(original, expanded, validation, tests, analysis), encoding="utf-8")
    build_pdf(args.pdf, original, expanded, validation, tests, analysis)
    receipt = Path(args.evidence).with_name("expanded_publication.json")
    publication.update(markdown_sha256=sha256(markdown), pdf_sha256=sha256(args.pdf),
                       validation_sha256=sha256(args.validation), tests_sha256=sha256(args.tests),
                       comparison_figure_sha256=sha256(ROOT / "results/figures/expanded_comparison.png"))
    receipt.write_text(json.dumps(publication, indent=2)+"\n")
    from pypdf import PdfReader
    pages = len(PdfReader(args.pdf).pages)
    print(json.dumps({"markdown": str(markdown), "pdf": args.pdf, "pages": pages,
                      "additional_datasets": 20, "additional_gpu_runs": 120, "combined_datasets": 23,
                      "combined_gpu_runs": 138, "tests": tests, "primary_comparisons": analysis["comparisons"],
                      "low_rank_new_runs": analysis["expanded_low_rank"], "published_evidence": publication}, indent=2))


if __name__ == "__main__":
    main()
