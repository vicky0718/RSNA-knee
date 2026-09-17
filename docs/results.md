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
