# NeurIPS 2026 Atlanta workshop assessment

Checked September 17, 2026 (UTC). This is a venue assessment and manuscript preparation record, not a submission or acceptance notice.

## Recommendation

Target the **non-archival track of the 1st Workshop on Physical World AI: Geometry, Characteristics, and Multimodal Sensing (PhysWorldAI)**. It is the most plausible time-compatible Atlanta option verified in this search. Its official call advertises a **September 29--October 29, 2026** non-archival submission window and says the venue will open soon. The archival deadline, September 9, has passed. The call allows up to eight main-text pages or four-page extended abstracts, excludes references/appendices from those limits, requires NeurIPS 2026 style, and uses double-blind review. It lists November 9 notification and November 19 camera-ready dates. [Workshop CFP](https://physworld-org.github.io/physworld.github.io/cfp/).

The [official NeurIPS workshop announcement](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/) lists PhysWorldAI in Atlanta. Atlanta workshops are December 12--13; the workshop has not fixed its day on its own site. [Conference dates](https://neurips.cc/Conferences/2026/Dates), [workshop home](https://physworld-org.github.io/physworld.github.io/).

**Availability distinction:** this is an announced future window, not a verified currently open submission form. The linked [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PhysWorldAI) exists. Its public group metadata still advertises an August 1 start and September 12, 2026, 23:59 UTC deadline, consistent with an earlier round rather than the forthcoming non-archival window. The non-archival form's availability was not established. The workshop page does not specify a non-archival cutoff timezone, and its November 9 notification date is later than the conference's mandatory September 29 workshop notification date. Confirm the operative track, timezone, and dates with the organizers when the new form appears. The dates above should not be silently interpreted as AoE. [General workshop guidance](https://neurips.cc/Conferences/2026/CallForWorkshops).

## Why this paper fits, and where it is weak

The workshop explicitly includes sensing, benchmarks, and evaluation protocols. This paper can contribute an auditable task-construction and evaluation protocol, with sensing-related public features and controlled synthetic dynamics. The useful physical-world connection is preservation of observation budgets, source groups, and forecast horizons.

Fit is **moderate**, not strong enough to promise acceptance. The experiments contain no action-conditioned world model, raw multimodal fusion, physical property estimation, robot interaction, or sim-to-real evaluation. Several public datasets are general classification benchmarks. The manuscript preserves those limits and reports all 23 datasets; it does not relabel ordinary classification as physical reasoning. The most defensible framing is a methods-and-evaluation pilot with mixed and negative findings.

## Alternatives checked

| Workshop in Atlanta | Posted submission deadline | Assessment for this project |
|---|---|---|
| [AI & Science: Evolution or Extinction?](https://aiscik.github.io/) | September 7, 2026, 23:59 AoE | Stronger connection to scientific evaluation integrity, but the posted deadline has passed. Its policy also excludes substantially AI-generated submissions, so this AI-assisted draft would require careful eligibility assessment and substantive human authorship before considering that venue. |
| [Interpretability for Discovery](https://interpretability4discovery.github.io/) | Extended to September 2, 2026, 23:59:59 AoE | Closed by posted date; the current study does not establish discovered scientific knowledge or validated explanations. |
| [DynaFront](https://sites.google.com/view/dynafrontneurips26) | Extended to September 4, 2026, AoE | Closed by posted date; its optimization/sampling/game dynamics scope is more theoretical than this prototype's dynamics datasets. |
| [World Models for High-Stakes Health](https://wmhs-neurips.github.io/WMHS/) | Extended to September 15, 2026, AoE | Deadline passed; a breast-cancer classification dataset does not establish patient trajectories, clinical-trial simulation, or intervention-aware reasoning. |
| [ML x OR](https://mlxor-2026.github.io/) | August 31, 2026, AoE | Deadline passed; no operational optimization or uncertainty-aware decision experiment is implemented here. |
| [Resource-Aware Agentic AI](https://resource-aware-workshop.github.io/) | August 29, 2026, AoE | Deadline passed; this experiment is not a resource-adaptive agent evaluation. |

This is a targeted assessment of plausible matches from the official Atlanta list, not a claim that every workshop and possible organizer exception was exhaustively checked. A past deadline is not evidence that organizers will accept late papers.

## Manuscript positioning

**Title:** JEPA-FORGE: Auditable Context-Target Selection for Predictive Representations.

The manuscript describes the implemented task contracts, development-only selector, fixed-horizon candidates, GPU experiments, matched baselines, and replay checks. It reports 273 new candidate-training runs on 23 datasets and keeps the earlier 138 runs separate. The strongest empirical message is mixed utility and low covariance rank despite nonzero coordinate variance. It does not claim that automatic selection improved test performance over fixed masks; that controlled experiment was not performed.

The draft is anonymous, uses the official 2026 style's workshop layout, and includes references and supplementary details. It contains no author-linked repository URL or private proposal material. The source archive is intended for Overleaf import. The companion evidence package provides code, measured results, configurations, and reproduction instructions; it does not contain the locally retained large checkpoint collection or private Git history.

## Before submission

1. Confirm the non-archival submission window and exact timezone in the live workshop portal; retain the eight-page long-paper or four-page abstract limit shown in the operative call.
2. Have all human authors review the scientific claims, source-linked references, figures, authorship order, affiliations, and AI-use disclosure. The current document is a draft and has not been submitted. The [NeurIPS author handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) makes human authors responsible for correctness and excludes AI systems from authorship.
3. Confirm the workshop's checklist requirements and supplement size/format once its non-archival form opens. The manuscript includes a factual reproducibility statement; it does not assert that every main-track administrative requirement automatically applies to the workshop.
4. For stronger physical-world evidence, prioritize real raw sensor trajectories and repeated group splits, followed by a same-split fixed-mask comparison and covariance-regularization ablation. These are proposed experiments, not additions to the reported results.

## Optional inquiry draft - not sent

Hello PhysWorldAI organizers,

We are preparing a paper on auditable context-target selection for predictive representations. It studies 23 public/synthetic datasets, including radar/sonar/activity/satellite features and grouped synthetic dynamics, with fixed observation budgets, validation-only selection, and mixed baseline results. It does not include robotics or action-conditioned world-model experiments.

Your call lists a September 29--October 29 non-archival window. Could you confirm the submission portal, exact cutoff timezone, current format requirements, and whether this evaluation-protocol contribution is in scope? We also noticed that the listed notification date is later than the general conference workshop guidance and would appreciate confirmation of the applicable schedule.

Thank you.
