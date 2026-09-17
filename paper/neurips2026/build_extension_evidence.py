"""Build manuscript/report evidence only from completed, validated extension runs."""
from __future__ import annotations
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"scripts"))
from validate_extension import validate_payload

NAMES={"wine":"Wine","digits":"Digits","breast_cancer":"WDBC","ionosphere":"Ionosphere","sonar":"Sonar",
       "semeion":"Semeion","letter":"Letter","pendigits":"PenDigits","satimage":"Satimage","har":"HAR features",
       "dry_bean":"Dry Bean","isolet":"ISOLET","synthetic_sensors":"Oscillatory sensors","synthetic_lorenz":"Lorenz",
       "synthetic_mackey_glass":"Delayed feedback","synthetic_switching_var":"Switching VAR","synthetic_chirp_seasonal":"Chirp/seasonal",
       "synthetic_coupled_oscillators":"Coupled oscillators","synthetic_nonlinear_ar":"Nonlinear AR",
       "synthetic_nonlinear_multiview":"Nonlinear multiview","synthetic_hierarchical_multiclass":"Hierarchical classes",
       "synthetic_sparse_interactions":"Sparse interactions","synthetic_manifold_nuisance":"Manifold/nuisance",
       "mhealth_activity":"MHEALTH activity","mhealth_forecast":"MHEALTH forecast",
       "pamap2_activity":"PAMAP2 activity","pamap2_forecast":"PAMAP2 forecast","synthetic_motion_video":"Synthetic motion video"}
METHODS={"jepa":"JEPA selection","jepa_fixed_budget":"Fixed mask, matched budget","jepa_fixed_short":"Fixed mask, 30 epochs",
         "supervised":"Supervised encoder","tabm":"TabM","linear":"Raw linear","extra_trees":"Extra Trees","catboost":"CatBoost",
         "filter_linear":"Filter + linear","filter_extra_trees":"Filter + Extra Trees",
         "embedded_linear":"Embedded + linear","embedded_extra_trees":"Embedded + Extra Trees"}


def load_frozen(stem):
    raw=gzip.decompress((ROOT/f"results/{stem}.json.gz").read_bytes())
    payload=json.loads(raw)
    validation=json.loads((ROOT/f"results/{stem}_validation.json").read_text())
    assert validation["status"]=="passed"
    assert hashlib.sha256(raw).hexdigest()==validation["result_uncompressed_sha256"]
    counts=validate_payload(payload,stem=="vision_transfer")
    return payload,validation,counts


def summarize(payload):
    grouped={}
    for result in payload["final_evaluations"]:
        name=result["dataset"]; metric="macro_f1" if result["classification"] else "rmse"
        if name not in grouped: grouped[name]={"classification":result["classification"],"metric":metric,"methods":defaultdict(list),"selections":defaultdict(list)}
        for row in result["methods"]:
            grouped[name]["methods"][row["method"]].append(row["test"][metric])
            grouped[name]["selections"][row["method"]].append({"split_seed":result["split_seed"],"mask":row["task"],"epoch":row.get("epoch")})
    for record in grouped.values():
        record["methods"]={k:{"values":v,"mean":float(np.mean(v)),"sd":float(np.std(v,ddof=1))} for k,v in record["methods"].items()}
    return grouped


def cell(row,method):
    if method not in row["methods"]: return "--"
    r=row["methods"][method]
    return f"${r['mean']:.3f}\\pm{r['sd']:.3f}$"


def rows_tex(path, rows):
    # The surrounding tabular owns the final line break before bottomrule.
    path.write_text("\\\\\n".join(" & ".join(row) for row in rows)+"\n")


def win_counts(summary,method,classification=None):
    wins=ties=losses=0
    for r in summary.values():
        if method not in r["methods"] or (classification is not None and r["classification"]!=classification): continue
        diff=r["methods"]["jepa"]["mean"]-r["methods"][method]["mean"]
        if not r["classification"]: diff=-diff
        wins+=diff>0; ties+=diff==0; losses+=diff<0
    return {"wins":int(wins),"ties":int(ties),"losses":int(losses),"total":int(wins+ties+losses)}


def win_cell(summary, method, classification):
    counts = win_counts(summary, method, classification)
    return f"{counts['wins']}/{counts['total']}" if counts['total'] else "--"


def main():
    extension,ev,counts=load_frozen("extension_benchmark")
    vision,vv,vcounts=load_frozen("vision_transfer")
    summary=summarize(extension); vsummary=summarize(vision)
    gen=HERE/"generated";fig=HERE/"figures"
    gen.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    comparisons={m:win_counts(summary,m) for m in METHODS if m!="jepa"}
    class_methods=["jepa","jepa_fixed_budget","supervised","tabm","extra_trees","catboost"]
    forecast_methods=class_methods[:-1]
    for category,flag,methods in [("extension_classification",True,class_methods),("extension_forecasting",False,forecast_methods)]:
        rows_tex(gen/f"{category}.tex",[[NAMES[n]]+[cell(r,m) for m in methods] for n,r in summary.items() if r["classification"]==flag])
    rows_tex(gen/"extension_sensors.tex",[[NAMES[n]]+[cell(r,m) for m in forecast_methods] for n,r in summary.items() if n.startswith(("mhealth_","pamap2_"))])
    task_label=lambda n,r: NAMES[n]+(r" ($\uparrow$)" if r["classification"] else r" ($\downarrow$)")
    rows_tex(gen/"extension_features.tex",[[task_label(n,r)]+[cell(r,m) for m in ["linear","filter_linear","embedded_linear","filter_extra_trees","embedded_extra_trees"]] for n,r in summary.items()])
    rows_tex(gen/"extension_policy.tex",[[task_label(n,r)]+[cell(r,m) for m in ["jepa","jepa_fixed_short","jepa_fixed_budget"]] for n,r in summary.items()])
    rows_tex(gen/"extension_vision.tex",[[NAMES[n]]+[cell(r,m) for m in ["jepa","supervised","extra_trees", "vjepa_official" if n=="synthetic_motion_video" else "ijepa_official"]] for n,r in vsummary.items()])
    rows_tex(gen/"extension_counts.tex",[[METHODS[m]]+[win_cell(summary,m,c) for c in [True,False,None]] for m in ["jepa_fixed_budget","jepa_fixed_short","linear","extra_trees","supervised","tabm","catboost","filter_extra_trees","embedded_extra_trees"]])
    rows_tex(gen/"extension_masks.tex",[[NAMES[n]]+[r["selections"]["jepa"][i]["mask"].replace("_",r"\_") for i in range(3)] for n,r in summary.items()])
    neural=[c for s in extension["selections"] for c in s["lock"]["candidates"] if "checkpoint_path" in c]
    jepa=[c for c in neural if c["method"]=="jepa"]
    ranks=np.array([c["representation"]["effective_rank"] for c in jepa])
    stable=sum(len({s["mask"] for s in r["selections"]["jepa"]})==1 for r in summary.values())
    runs=sum(1 for c in neural if c["method"]!="jepa_fixed_budget")+len(extension["selections"])
    macros={"ExtTasks":len(summary),"ExtSplits":len(extension["selections"]),"ExtRuns":runs,
            "ExtCheckpoints":len(neural),"ExtPolicyWins":comparisons['jepa_fixed_budget']['wins'],
            "ExtShortWins":comparisons['jepa_fixed_short']['wins'],"ExtTreesWins":comparisons['extra_trees']['wins'],
            "ExtLinearWins":comparisons['linear']['wins'],"ExtSupervisedWins":comparisons['supervised']['wins'],
            "ExtTabMWins":comparisons['tabm']['wins'],"ExtCatBoostWins":comparisons['catboost']['wins'],
            "ExtRankFlags":int((ranks<2).sum()),"ExtRankRuns":len(ranks),"ExtRankMedian":f"{np.median(ranks):.2f}",
            "ExtStableTasks":stable,"ExtReplayError":f"{ev['cpu_gpu_max_absolute_error']:.2g}",
            "ExtPredictionReplays":ev['saved_test_predictions_recomputed'],"VisionReplayError":f"{vv['cpu_gpu_max_absolute_error']:.2g}",
            "VisionCheckpoints":vv['cpu_gpu_checkpoint_replays']}
    (gen/"extension_macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in macros.items())+"\n")
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    figure,axes=plt.subplots(1,2,figsize=(7,4.8),gridspec_kw={"width_ratios":[1.2,1]})
    for ax,classification in zip(axes,[True,False]):
        items=[(n,r) for n,r in summary.items() if r["classification"]==classification]
        means,sds=[],[]
        for name,r in items:
            a=np.asarray(r['methods']['jepa']['values']); b=np.asarray(r['methods']['jepa_fixed_budget']['values'])
            diff=a-b if classification else 100*(b-a)/b
            means.append(diff.mean());sds.append(diff.std(ddof=1))
        yy=np.arange(len(items));ax.axvline(0,color='0.5',lw=.7)
        ax.errorbar(means,yy,xerr=sds,fmt='o',markersize=3,color='#1b627a',elinewidth=.7,capsize=2)
        ax.set_yticks(yy,[NAMES[n] for n,_ in items]);ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7)
        ax.set_xlabel('Macro-F1 difference' if classification else 'RMSE reduction (%)')
        ax.set_title('Classification' if classification else 'Forecasting')
        ax.grid(axis='x',alpha=.2)
    figure.suptitle('Selected masks versus a fixed mask at the same training budget',fontsize=10)
    figure.tight_layout(rect=(0,0,1,.96))
    figure.savefig(fig/"extension_policy.pdf",metadata={"Title":"Repeated-split matched-budget mask comparison","Author":"","Creator":"JEPA-FORGE","CreationDate":None,"ModDate":None})
    plt.close(figure)
    evidence={"extension":summary,"vision":vsummary,"comparisons":comparisons,"macros":macros,
              "validation":ev,"vision_validation":vv,"rank_median":float(np.median(ranks)),
              "source_payload_hashes":{"extension":ev["result_uncompressed_sha256"],"vision":vv["result_uncompressed_sha256"]}}
    (gen/"extension_evidence.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    report=["# Repeated-split extension report", "", "Completed MPS experiments with independent baseline selection. Results are descriptive split means and sample standard deviations; three overlapping partitions do not support population-level significance claims.","",
            f"The main extension has {len(summary)} tasks on 25 underlying datasets, {len(extension['selections'])} task/split evaluations and {runs} neural training trajectories. It adds four tasks from two raw sensor corpora. The transfer study adds three subset/toy-video tasks; its two image datasets overlap the original collection.","",
            "## Main comparisons","","Baseline | JEPA wins | Ties | JEPA losses","---|---:|---:|---:"]
    for m,r in comparisons.items(): report.append(f"{METHODS[m]} | {r['wins']}/{r['total']} | {r['ties']} | {r['losses']}")
    report += ["","## Per-task results","","Metrics are macro F1 (higher is better) for classification and original-unit RMSE (lower is better) for forecasting. Every method chooses its own mask on validation.","",
               "Task | Metric | JEPA | Fixed matched | Supervised | TabM | Extra Trees | CatBoost","---|---|---:|---:|---:|---:|---:|---:"]
    for n,r in summary.items():
        values=[f"{r['methods'][m]['mean']:.6f} +/- {r['methods'][m]['sd']:.6f}" if m in r['methods'] else 'n/a' for m in class_methods]
        report.append(f"{NAMES[n]} | {r['metric']} | "+" | ".join(values))
    report += ["","## Official-model transfer","","Task | JEPA | Supervised | Extra Trees | Official pretrained","---|---:|---:|---:|---:"]
    for n,r in vsummary.items():
        methods=['jepa','supervised','extra_trees','vjepa_official' if n=='synthetic_motion_video' else 'ijepa_official']
        report.append(NAMES[n]+" | "+" | ".join(f"{r['methods'][m]['mean']:.6f} +/- {r['methods'][m]['sd']:.6f}" for m in methods))
    report += ["","Official backbones are frozen, externally pretrained transfer baselines. Their pretraining cost, sizes and resizing are not matched to the scratch pilot. Video evidence is synthetic and uses eight-frame inference; no natural-video or official benchmark reproduction is claimed.","",
               "## Validation and reproduction","",f"{ev['cpu_gpu_checkpoint_replays']} main-study checkpoint snapshots and {vv['cpu_gpu_checkpoint_replays']} transfer-study scratch snapshots were replayed on eight training examples each on CPU and MPS. Maximum absolute differences were {ev['cpu_gpu_max_absolute_error']:.8g} and {vv['cpu_gpu_max_absolute_error']:.8g}; atol/rtol were 5e-4. All {ev['saved_test_predictions_recomputed']} main and {vv['saved_test_predictions_recomputed']} transfer saved prediction files reproduced the reported metrics.","",
               "See docs/EXTENSION_PROTOCOL.md and docs/VISION_TRANSFER_PROTOCOL.md for exact budgets, grouping, sampling and limits. Compressed evidence and validation reports are under results/. The paper sources and result-derived tables are under paper/neurips2026/.","",
               "Model selection remains supervised through development labels. Raw sensor forecasts exclude null/transition windows using source annotations; PAMAP2 is decimated without antialias filtering. Most prior datasets are reused, only one neural initialization is used per split, and broad foundation-model tuning remains outside this bounded experiment."]
    (ROOT/"EXTENSION_REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"comparisons":comparisons,"macros":macros},indent=2))


if __name__=="__main__": main()
