"""Derive manuscript tables and figures from the frozen selection experiment."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_selection import validate_payload

NAMES = {
    "wine": "Wine", "digits": "Digits", "synthetic_sensors": "Sensor oscillator",
    "breast_cancer": "Breast Cancer", "ionosphere": "Ionosphere", "sonar": "Sonar",
    "semeion": "Semeion", "letter": "Letter", "pendigits": "PenDigits",
    "satimage": "Satimage", "har": "HAR", "dry_bean": "Dry Bean", "isolet": "ISOLET",
    "synthetic_lorenz": "Lorenz", "synthetic_mackey_glass": "Delayed dynamics",
    "synthetic_switching_var": "Switching VAR", "synthetic_chirp_seasonal": "Chirp-seasonal",
    "synthetic_coupled_oscillators": "Coupled oscillators", "synthetic_nonlinear_ar": "Nonlinear AR",
    "synthetic_nonlinear_multiview": "Multiview factors", "synthetic_hierarchical_multiclass": "Hierarchical classes",
    "synthetic_sparse_interactions": "Sparse interactions", "synthetic_manifold_nuisance": "Manifold + nuisance",
}
METHODS = ["jepa_context_linear", "raw_context_linear", "raw_context_extra_trees", "random_encoder_linear", "raw_context_pca"]

def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

def ranges(values):
    """Compact exact ordered coordinates without changing the task."""
    out=[]; values=list(values); i=0
    while i < len(values):
        j=i
        while j+1 < len(values) and values[j+1] == values[j]+1: j+=1
        out.append(str(values[i]) if j==i else f"{values[i]}--{values[j]}")
        i=j+1
    return ", ".join(out)

def main():
    raw=gzip.decompress((ROOT/"results/selection_benchmark.json.gz").read_bytes())
    data=json.loads(raw); config=json.loads((ROOT/"configs/selection.json").read_text())
    validation=json.loads((ROOT/"results/selection_validation.json").read_text())
    assert hashlib.sha256(raw).hexdigest()==validation["results_sha256"]
    validate_payload(data,config)
    (HERE/"generated").mkdir(parents=True,exist_ok=True)
    (HERE/"figures").mkdir(parents=True,exist_ok=True)
    selections={s["dataset"]:s["lock"] for s in data["selections"]}
    rows=[]; diagnostics=[]; selected_flags=0; param=[]
    for final in data["final_evaluations"]:
        name=final["dataset"]; lock=selections[name]
        metric="macro_f1" if lock["metric"]=="macro_f1" else "rmse"
        row={"dataset":name,"display":NAMES[name],"metric":metric,"task":final["task"],"split_sizes":final["split_sizes"],"methods":{}}
        for method in final["runs"][0]["evaluation"]:
            vals=[next(v for v in r["evaluation"] if v["method"]==method["method"])["test"][metric] for r in final["runs"]]
            row["methods"][method["method"]]={"mean":float(np.mean(vals)),"sd":float(np.std(vals,ddof=1)),"values":vals}
        row["input_dim"]=lock["candidates"][0]["runs"][0]["training"]["model_spec"]["input_dim"]
        row["modality"]=lock["candidates"][0]["runs"][0]["training"]["model_spec"]["modality"]
        rows.append(row)
        for index,candidate in enumerate(lock["candidates"]):
            for run in candidate["runs"]:
                training=run["training"]; d=training["final_train_context_diagnostics"]
                diagnostics.append({"dataset":name,"candidate":candidate["task"]["name"],"seed":run["seed"],"selected":index==lock["selected_index"],**d})
                selected_flags+=int(index==lock["selected_index"] and d["heuristic_collapse_flag"])
                param.append(training["trainable_parameters"])
    groups={}
    for metric in ["macro_f1","rmse"]:
        subset=[r for r in rows if r["metric"]==metric]; direction=1 if metric=="macro_f1" else -1
        groups[metric]={"datasets":len(subset),"wins":{m:sum(direction*(r["methods"][METHODS[0]]["mean"]-r["methods"][m]["mean"])>1e-12 for r in subset) for m in METHODS[1:]}}
    ranks=np.array([d["effective_rank"] for d in diagnostics]); std=np.array([d["mean_std"] for d in diagnostics])
    evidence={"source_sha256":hashlib.sha256(raw).hexdigest(),"groups":groups,"candidate_runs":len(diagnostics),"low_rank_flags":sum(d["heuristic_collapse_flag"] for d in diagnostics),"rank_below_two":int((ranks<2).sum()),"mean_std_below_threshold":int((std<.01).sum()),"selected_flags":selected_flags,"median_effective_rank":float(np.median(ranks)),"rank_quartiles":np.quantile(ranks,[.25,.75]).tolist(),"trainable_parameters_range":[min(param),max(param)],"rows":rows,"diagnostics":diagnostics}
    assert evidence["candidate_runs"]==273 and evidence["low_rank_flags"]==198 and len(rows)==23
    assert groups["macro_f1"]["wins"]["raw_context_linear"]==11 and groups["rmse"]["wins"]["raw_context_linear"]==2
    (HERE/"generated/evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    def cell(r,m):
        x=r["methods"][m];return f"{x['mean']:.3f} $\\pm$ {x['sd']:.3f}"
    for metric,filename in [("macro_f1","classification"),("rmse","forecasting")]:
        lines=[]
        for r in rows:
            if r["metric"]!=metric:continue
            lines.append(tex(r["display"])+" & "+str(len(r["task"]["context"]))+" & "+" & ".join(cell(r,m) for m in METHODS[:3])+r" \\")
        lines[-1] = lines[-1][:-2]
        (HERE/f"generated/{filename}.tex").write_text("\n".join(lines)+"\n")
    inventory=[];other=[];contexts=[];candidate_lines=[]
    for r in rows:
        sp=r["split_sizes"];lock=selections[r["dataset"]]
        inventory.append(f"{tex(r['display'])} & {r['input_dim']} & {sp['train']}/{sp['val']}/{sp['test']} & {tex(r['task']['name'].replace('_',' '))}" + r" \\")
        ms=["random_encoder_linear","raw_context_pca","full_raw_linear_reference" if r["metric"]=="macro_f1" else "persistence"]
        other.append(tex(r["display"])+" & "+" & ".join(cell(r,m) for m in ms)+r" \\")
        contexts.append(r"\paragraph{"+tex(r["display"])+"} "+r"Context: \texttt{"+ranges(r["task"]["context"])+"}. Target: "+r"\texttt{"+ranges(r["task"]["target"])+"}.\n")
        for i,c in enumerate(lock["candidates"]):
            score=c["mean_validation_score"]*(1 if r["metric"]=="macro_f1" else -1)
            candidate_lines.append(f"{tex(r['display'])} & {tex(c['task']['name'].replace('_',' '))} & {score:.5f} & {c['std_validation_score']:.5f} & {'yes' if i==lock['selected_index'] else ''}" + r" \\")
    for name,lines in [("inventory",inventory),("other_baselines",other),("contexts",contexts),("candidates",candidate_lines)]:
        if name in {"inventory", "other_baselines"}: lines[-1] = lines[-1][:-2]
        (HERE/f"generated/{name}.tex").write_text("\n".join(lines)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
    fig,ax=plt.subplots(1,2,figsize=(8.1,4.9),gridspec_kw={"width_ratios":[1.16,1]})
    for a,metric in zip(ax,["macro_f1","rmse"]):
        subset=[r for r in rows if r["metric"]==metric];y=np.arange(len(subset))
        for m,color,marker,label in [("raw_context_linear","#186b94","o","Raw linear"),("raw_context_extra_trees","#b45430","s","Extra Trees")]:
            vals=[]
            for r in subset:
                j=r["methods"][METHODS[0]]["mean"]; b=r["methods"][m]["mean"]
                vals.append(j-b if metric=="macro_f1" else 100*(b-j)/b)
            a.scatter(vals,y,s=25,color=color,marker=marker,label=label,zorder=3)
        a.axvline(0,color="#777777",lw=.9);a.set_yticks(y,[r["display"] for r in subset]);a.invert_yaxis();a.grid(axis="x",alpha=.2)
        a.set_title("Classification (16 datasets)" if metric=="macro_f1" else "Forecasting (7 datasets)",fontsize=10)
        a.set_xlabel("JEPA minus baseline macro F1" if metric=="macro_f1" else "RMSE reduction relative to baseline (%)",fontsize=8)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(w_pad=2.4, rect=[0, .07, 1, 1])
    fig.savefig(HERE/"figures/comparisons.pdf",bbox_inches="tight",metadata={"Author":"","Title":"Matched-context baseline comparisons"});plt.close(fig)
    histories=[]
    for s in data["selections"]:
        for c in s["lock"]["candidates"]:
            for r in c["runs"]:
                histories.append([e["train_context_diagnostics"]["effective_rank"] for e in r["training"]["training_history"]])
    h=np.array(histories);epochs=np.arange(1,61)
    fig,ax=plt.subplots(1,2,figsize=(7,2.5))
    ax[0].fill_between(epochs,np.quantile(h,.25,axis=0),np.quantile(h,.75,axis=0),color="#c6dce8",label="Interquartile range")
    ax[0].plot(epochs,np.median(h,axis=0),color="#186b94",label="Median")
    ax[0].axhline(2,color="#b45430",ls="--",lw=1,label="Flag threshold")
    ax[0].set(xlabel="Training epoch",ylabel="Training covariance effective rank");ax[0].legend(fontsize=7,frameon=False)
    ax[1].scatter(std,ranks,c=["#b45430" if d["heuristic_collapse_flag"] else "#186b94" for d in diagnostics],s=10,alpha=.55)
    ax[1].axhline(2,color="#b45430",ls="--",lw=1)
    ax[1].set(xlabel="Final mean coordinate standard deviation",ylabel="Final effective rank")
    fig.tight_layout()
    fig.savefig(HERE/"figures/rank_diagnostics.pdf",bbox_inches="tight",metadata={"Author":"","Title":"Rank diagnostics for all 273 candidate runs"});plt.close(fig)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['rows','diagnostics']},indent=2))

if __name__=="__main__":main()
