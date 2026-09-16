#!/usr/bin/env python3
"""Scientific comparison figure derived from completed, fixed primary policies."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
sys.path.insert(0, str(ROOT/'scripts'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_expanded_report import analyses, label


def main():
    original = json.loads((ROOT/'results/benchmark.json').read_text())
    expanded = json.loads((ROOT/'artifacts/expanded/results.json').read_text())
    if expanded['status'] != 'complete' or len(expanded['runs']) != 120:
        raise ValueError('Complete expanded matrix required')
    result = analyses(original, expanded)
    panels = [[], []]
    for audit in result['primary']:
        key = (audit['dataset'], audit['task'])
        j = result['rows'][(*key, 'jepa_context_linear')]
        b = result['rows'][(*key, 'raw_context_linear')]
        classification = j['metric'] == 'macro_f1'
        gain = 100*(j['mean']-b['mean']) if classification else 100*(b['mean']-j['mean'])/b['mean']
        std = 100*j['std'] if classification else 100*j['std']/b['mean']
        panels[0 if classification else 1].append((label(key[0]),gain,std))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(13.5,7.4),gridspec_kw={'width_ratios':[1.25,1]})
    for ax, rows, title, xlabel in zip(axes,panels,
        ['Classification: 14 new datasets','Forecasting: 6 new synthetic systems'],
        ['Macro-F1 change (percentage points)','RMSE reduction relative to raw linear (%)']):
        names, values, errors = zip(*rows)
        y = np.arange(len(names))
        ax.barh(y, values, xerr=errors, height=.67, color=['#087E8B' if value>=0 else '#BE642B' for value in values], capsize=2)
        ax.set_yticks(y,names)
        ax.invert_yaxis()
        ax.axvline(0,color='#203548',linewidth=.8)
        ax.grid(axis='x',alpha=.17)
        ax.set_axisbelow(True)
        ax.set_title(title,fontweight='bold',pad=13)
        ax.set_xlabel(xlabel,labelpad=10)
    fig.suptitle('JEPA relative to the matched raw-context linear baseline',fontsize=17,fontweight='bold',x=.51,y=.975)
    fig.text(.5,.022,'Positive values favor JEPA. Primary policies fixed before evaluation. Bars: three-seed means; whiskers: one model-seed SD, not confidence intervals.\nRMSE is compared within each system; percentages do not average across systems. Extra Trees and other baselines appear in the report tables.',ha='center',fontsize=9,color='#46586A')
    fig.tight_layout(rect=(0,.065,1,.94),w_pad=3)
    path=ROOT/'results/figures/expanded_comparison.png'
    path.parent.mkdir(exist_ok=True,parents=True)
    fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
