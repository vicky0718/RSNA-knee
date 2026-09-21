# Measured results

A running log of things we checked rather than assumed. Every row here is
reproducible from a script in `bin/`. Claims that did not survive measurement
are kept, not deleted — knowing what we got wrong is how the next claim gets
checked.

---

## 1. Label quality — the public table is strong (2026-09-15)

`bin/score_labels.py`, macro AUC against the 58 gold studies.

| source | macro AUC |
|---|---|
| `llm_labels_v4_blend.csv` | **0.893** |
| `llm_labels_v2.csv` | 0.887 |
| `llm_labels_full.csv` | 0.878 |
| `labels_llm_gpt56sol.csv` | 0.835 |
| our rule reader, calibrated | 0.748 |

Gap from ours to the best: **+0.144, 95% CI [+0.102, +0.188]**.

Per-target, the public table is weak in only four places — **Synovitis 0.790,
Fracture 0.793, Lateral OA 0.833, Contusion 0.860** — and reads everything else
at 0.88–0.99 (ACL 0.987, MCL 0.968, Medial Meniscus 0.948).

Blending our reader in lowers the score monotonically (0.893 → 0.879 at 40%
weight) despite being genuinely decorrelated (rank correlation 0.53 against the
public tables' 0.97–0.995 among themselves). **Decorrelation without accuracy is
worth nothing.**

> **Correction.** An earlier version of this analysis reported that the
> published 0.893 "does not reproduce" and put the best public table at 0.819.
> That was wrong: the public tables hold calibrated probabilities, and
> binarising them at 0.5 to measure agreement discards the ranking the metric
> rewards. Scored as AUC, the published figure reproduces exactly.

**Consequence:** adopt the public table as the label baseline. Point any LLM
reader at the four weak findings, not at the table as a whole.

---

## 2. The shared-report leak is boilerplate, and costs ~nothing (2026-09-15)

`bin/measure_leak.py`. A TF-IDF character n-gram model over the report text,
predicting the public labels, scored out-of-fold under two splits.

| split | OOF macro AUC vs public labels | vs the 58 gold |
|---|---|---|
| random | 0.9292 | 0.7985 |
| report-grouped | 0.9290 | 0.7992 |

**Inflation from ignoring the report leak: +0.0002 macro AUC.** Even restricted
to the 250 studies that share a report with another study — the only rows where
the two splits differ — it is 0.9988 against 0.9975.

The reason is visible in the clusters themselves. They are **templates, not
repeat scans**: 37 studies sharing a Turkish "everything normal" report, 21
sharing `Sin anomalías`, 21 sharing a Hoffa fat-pad impingement line. Clustered
reports have a median length of 205 characters against the corpus median of 977.
Different patients, identical boilerplate.

> **Consequence.** The report-duplicate guard stays — it costs 0.0002 and
> protects against repeat scans that terse reports would hide — but it is no
> longer a pillar of the validation argument, and this document should stop
> being cited as if it were. **The scanner/site leak is the untested one**, and
> it is the one worth the effort: scanner identity is in the DICOM headers, so
> it needs a pass on Kaggle (`bin/build_folds.py --scanner-keys`).

Caveat on scope: this measures what the report leak is worth to a model that can
*read the report*, which is an upper bound on the image pipeline's exposure, not
its exact figure.

---

## 3. Data audit (2026-09-15)

`bin/audit_data.py`. 4,407 studies, 24,371 series, exactly **58** with all twelve
labels, every study has a report.

Slot coverage reproduces the public pipeline's figures to three decimals:
AX_FS 1.000, SAG_NOFS 0.968, COR_FS 0.964, SAG_FS 0.942, COR_NOFS 0.773,
**AX_NOFS 0.194**.

`Fluid_Sensitive` and `Fat_Suppression` are **identical for all 24,371 series**,
despite the data description warning they "are not necessarily equivalent" — so
a slot keyed on both can be unfillable, which is a bug we shipped once.

---

## 4. The scanner leak is worth +0.0295 macro AUC (2026-09-17)

`bin/measure_leak.py`, same TF-IDF text model, three splits:

| split | OOF macro AUC vs public labels | vs the 58 gold |
|---|---|---|
| random | 0.9292 | 0.7985 |
| report-grouped | 0.9290 | 0.7992 |
| **scanner-grouped** | **0.8997** | **0.7761** |

| guard | inflation it removes |
|---|---|
| report | +0.0002 |
| **scanner** | **+0.0295** |

**A random split reads ~0.03 too high — three times the entire gold gap** (0.951 cut against a
0.941 public plateau). Any team validating without a scanner guard is choosing models on a number
that is wrong by more than the prize is wide.

**Why it leaks.** Prevalence varies enormously by machine, so recognising the scanner predicts the
label prior without learning any pathology:

| scanner | n | PF OA | Synovitis | Medial OA | ACL |
|---|---|---|---|---|---|
| Philips Ingenia 3T | 450 | 0.22 | 0.01 | 0.24 | 0.14 |
| Siemens MAGNETOM Vida 3T | 353 | 0.67 | 0.04 | 0.59 | 0.26 |
| SIEMENS Aera 1.5T | 400 | 0.38 | 0.10 | 0.36 | 0.32 |
| Philips Achieva dStream 3T | 141 | 0.33 | 0.28 | 0.25 | 0.43 |
| GE SIGNA Artist 1.5T | 208 | 0.58 | 0.22 | 0.38 | 0.09 |

The same effect shows up in the fold statistics: per-finding prevalence spread across folds is
0.04 under report-grouping and **0.18** under scanner-grouping.

**The two guards cannot compose.** Report boilerplate bridges scanners, so the transitive closure
of both collapses to 37 groups with the largest holding 42.5% of studies. `bin/build_folds.py`
takes `--guard report|scanner|both`; **scanner is the default**, because it is the one that
measures.

Caveat on scope, as in §2: this is a text model, which identifies the institution from reporting
style. An image model would identify the scanner from acquisition characteristics — plausibly more
easily, not less — but the exact magnitude for the image pipeline is still to be confirmed.

### Header extraction, for the record

`notebooks/kaggle/extract_headers.py` read one header per series for all 24,371 series in ~7
minutes (10.4 studies/s), zero unreadable. 59 distinct scanners; `Laterality` present on 79% of
series, `MagneticFieldStrength` on 95%, `SeriesDescription` on 94%. The same walk over ~1,322 test
studies costs ~2 minutes, so headers are not the constraint on the 9-hour budget. Pixel decoding
is still unmeasured.

---

## 5. The public stack's own diagnostics agree with bet #2 (2026-09-18)

Reproducing the public 0.941 pipeline privately (`knee-public-0941-repro`, all sixteen sources
attached) produced a valid `submission.csv` — rank values, not the 0.5 fallback its error path
writes. Along the way it printed its author's own arm-agreement diagnostic, which is worth
recording because it independently names where LB-fitted fusion weights are dangerous:

> These findings have two arms that already agree above 0.99, so their outer weight is close to a
> no-op no matter what it is set to: **Lateral Meniscus, PF OA, Synovitis, Baker's, Fracture**
>
> These findings have genuinely different arms, which is where an outer weight actually decides
> the ordering — and therefore where a weight fitted to the public split can do real damage
> privately: **ACL, MCL, Medial Meniscus, Medial OA, Lateral OA, Effusion, Contusion**

Note what this implies about probe #22, the LB-tuned routing ~350 teams share: it set Lateral
Meniscus to 1.00, discarding three of four stages for that column — and Lateral Meniscus is on the
*no-op* list, where the arms agree above 0.99 anyway. The headline change of that probe was
fitted to noise on a 30% split.

**Consequence for the OOF refit (§4, bet #2):** prioritise the seven findings where the arms
genuinely disagree. That is where an honestly fitted weight can gain, and where the crowd's
fitted-to-public weight can lose.

---

## 6. Plan 2: the public folds are not scanner-guarded, and per-finding routing is noise (2026-09-19)

`notebooks/kaggle/inspect_weights.py` read metadata out of every public weight bundle, looking for
a recoverable fold assignment. It found better than that.

**`v52_oof.csv` and `v52_e11_oof.csv` ship real out-of-fold predictions** — 4,407 rows, all twelve
targets, plus `fold` and `is_gold` columns. Two of the public RadImageNet arms therefore come with
their OOF and their fold assignment already computed. Several bundles also carry their own scores
(`weak_oof_auc`, `gold_oof_auc` ≈ 0.82–0.86).

### Their folds are random with respect to scanner

Cross-referencing the `fold` column against the scanner keys we extracted:

- **48 of 59 scanners span more than one fold.** The 11 that don't are singletons.
- Every large scanner is spread ~20% into each of the five folds:

| scanner | fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Philips Ingenia 3T | 0.22 | 0.20 | 0.18 | 0.16 | 0.23 |
| SIEMENS Aera 1.5T | 0.17 | 0.17 | 0.26 | 0.20 | 0.20 |
| MAGNETOM Vida 3T | 0.20 | 0.22 | 0.18 | 0.21 | 0.19 |

So the public pipeline's own OOF numbers carry the full scanner leak measured in §4. Their
reported 0.82–0.86 is inflated by roughly the +0.0295 we measured on a proxy model. **This is the
direct confirmation of bet #2 against the leading public pipeline, not against a stand-in.**

### Per-finding routing does not survive a scanner-grouped held-out fold

Blending the two arms (rank correlation 0.729 — genuinely different, well under the 0.97
rejection threshold), weights fitted on their OOF, selection validated on scanner-grouped
outer folds via `fuse.honest_gain`:

| fusion | macro AUC |
|---|---|
| best single arm (v52) | 0.8307 |
| flat 50/50 blend | 0.8459 |
| **OOF-fitted global weight** (v52 0.545 / v52_e11 0.455) | **0.8460** |
| per-finding routing, in-sample | 0.8474 |
| per-finding routing, **held-out** | **0.8393** |

**Per-finding routing loses to a single global weight by 0.0067 on folds it did not see**, while
looking 0.0014 better in-sample. That is the probe-#22 failure mode reproduced end to end on real
public OOF: twelve correlated columns, fitted to a split, flattering themselves.

Blending two decorrelated arms is worth **+0.015**; tuning per-finding weights on top is worth
**−0.007**. Almost all the fitted weight is a fair coin (0.545/0.455).

Scope: this tests the *inner* RadImageNet two-arm blend. Probe #22 tunes the *outer*
CoAtNet-versus-transformer routing, and we have no OOF for those arms, so this is strong
supporting evidence rather than a direct test of that specific weight set.

**Consequences.** Use one global blend weight unless per-finding routing demonstrably survives
`honest_gain` on scanner-grouped folds. Decorrelated arms, not more members, are where the gain
is. And the public checkpoints are not a dead end for honest work after all: two arms' OOF is
already in hand, at zero compute cost.

---

## 7. Which arms ship OOF, and what the rest reveal (2026-09-21)

Completing the Plan 2 sweep across every public bundle.

### OOF availability

| arm | OOF shipped? | what it carries |
|---|---|---|
| RadImageNet v52 | **yes** | `v52_oof.csv`, 4,407 rows + `fold` + `is_gold` |
| RadImageNet v52_e11 | **yes** | `v52_e11_oof.csv`, same shape |
| DINO 20-member | no | `manifest.json` gives each member's fold, but no predictions |
| A5 5-fold | no | each checkpoint carries `fold`, no predictions |
| CoAtNet / Raptor | no | `gold_auc` and a per-target `aucs` dict only |

So the honest refit is limited to the two RadImageNet arms (§6) unless we spend GPU time
regenerating OOF for the others. The DINO arm's fold-per-member *is* recorded, so its OOF is
reconstructible in principle — but it needs a pass over all 4,407 studies, and the study→fold
mapping for that arm is not published (only the RadImageNet arms' is). That makes it a real
compute commitment, not a free one.

### Correction: the fold models do not use site as a feature

The A5 configs carry `n_sites: 109`, which looked like site conditioning — a model built on the
very leak we measured. It is not: the same config says `meta: "none"` and `n_meta: 0`, so the
conditioning path exists in the code and is switched off. The hypothesis is withdrawn.

One useful fact survives it: the organisers' data has **109 sites**, finer than the 59 distinct
scanner keys we recovered from the headers. Our scanner guard is therefore coarser than the true
site structure, and a site-level guard would likely be stricter still.

### The stack is label-limited on the findings that matter

Scoring the public label table and the strongest CoAtNet arm against the *same* 58 gold studies:

| target | label table | CoAtNet v5 | model − label |
|---|---|---|---|
| **Synovitis** | 0.790 | 0.797 | **+0.007** |
| Fracture | 0.793 | 0.917 | +0.124 |
| Lateral OA | 0.833 | 0.867 | +0.034 |
| Contusion | 0.860 | 0.928 | +0.068 |
| Effusion | 0.877 | 0.976 | +0.099 |
| **Lateral Meniscus** | 0.879 | 0.877 | **−0.002** |
| **PF OA** | 0.902 | 0.847 | **−0.055** |
| Medial OA | 0.932 | 0.977 | +0.045 |
| Baker's | 0.944 | 0.971 | +0.027 |
| Medial Meniscus | 0.948 | 0.957 | +0.009 |
| MCL | 0.968 | 0.980 | +0.012 |
| ACL | 0.987 | 0.965 | −0.022 |

Correlation between label quality and model performance: **0.691**. Mean margin: **+0.029**.

That splits the twelve findings into three regimes:

- **Label-limited** — the model has already reached its supervision and no architecture will move
  it: **Synovitis (+0.007), Lateral Meniscus (−0.002), Medial Meniscus (+0.009)**.
- **Model-limited** — the labels are better than the model is using: **PF OA (−0.055),
  ACL (−0.022)**.
- **Image carries more than the report** — vision genuinely adds: Fracture (+0.124),
  Effusion (+0.099), Contusion (+0.068).

**Synovitis is the single biggest drag on the macro average (0.79) and it is squarely
label-limited.** That revives bet #1 in a much narrower and more defensible form than "beat the
public label table": a reader aimed at Synovitis alone.

Caveat: the CoAtNet `gold_auc` values are almost certainly not out-of-fold — the 58 gold studies
were in the training pool under weak labels — so the model column is optimistic. That strengthens
the label-limited readings (optimistic and still at ceiling) and weakens the model-limited ones.
