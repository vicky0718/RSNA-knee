# RSNA Knee Abnormality Detection — Competition Brief

Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
(Host: Radiological Society of North America. Notes compiled 2026-09-14.)

## Task

Multimodal (MRI images + radiology report text) **multi-label classification**.
For each knee MRI *study*, predict the probability of each of 12 findings:

| Column | Finding |
|---|---|
| `ACL` | Anterior cruciate ligament injury |
| `MCL` | Medial collateral ligament injury |
| `Medial Meniscus` | Medial meniscus tear |
| `Lateral Meniscus` | Lateral meniscus tear |
| `Medial OA` | Osteoarthritis, medial tibiofemoral compartment |
| `Lateral OA` | Osteoarthritis, lateral tibiofemoral compartment |
| `PF OA` | Patellofemoral osteoarthritis |
| `Effusion` | Joint effusion |
| `Synovitis` | Synovial inflammation |
| `Baker's` | Baker's cyst |
| `Contusion` | Bone contusion / bruise |
| `Fracture` | Fracture |

First RSNA challenge pairing every imaging study with its original radiology report.

## Evaluation

- Metric: **macro-averaged ROC AUC** over the 12 targets (mean of per-label AUCs).
- Submission file `submission.csv`, row id column `StudyInstanceUID`:

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,...
```

- Public LB = 30% of test data; private = remaining 70%.
- Scores truncated to 3 decimals.

## Data

- `train.csv` — one row per study: `StudyInstanceUID`, `Report` (free text, **12 languages**,
  varies by institution), plus the 12 binary labels.
  **Only a small subset of training studies carry per-condition labels** — the reports are
  provided so labels for the rest can be derived (weak/pseudo labeling from text).
- `train_series.csv` — one row per series: `StudyInstanceUID`, `SeriesInstanceUID`,
  `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane`
  (`Sagittal`/`Coronal`/`Axial`). The two flags are correlated but not equivalent.
- `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` — one slice per file.
  Series typically 20–45 slices (median 30), long tail to a few hundred.
- `test.csv` / `test_series.csv` / `test_series/` — 3 example studies, replaced at scoring time by
  ~**1300 test studies**. **No `Report` field at test time** (text is train-only signal).
- `sample_submission.csv` — all label columns at 0.5.

DICOM notes: mixed transfer syntaxes (Explicit VR LE uncompressed, JPEG Lossless, JPEG 2000,
Implicit VR LE); intensities, orientations and resolutions vary; metadata stripped to an
allowlisted set of 86 tags. ~5,000+ training exams from 16 sites across 5 continents.

**Prevalence of abnormalities is not guaranteed to match across train / public LB / private LB.**

## Timeline (all 23:59 UTC)

| Date | Milestone |
|---|---|
| 2026-07-30 | Start |
| 2026-10-15 | Entry deadline / team merger deadline / notebook-publishing cutoff |
| 2026-10-22 | **Final submission deadline** |
| 2026-11-05 | Winners' requirements (code, video, method description) |
| Nov 2026 | Winners announced; recognized at RSNA 2026 (Nov 29 – Dec 3, Chicago) |

## Prizes — $77,000 total

Leaderboard: $9,000 / $7,000 / $6,500 / $6,000 / $5,500 / $5,000 / $5,000 / $5,000 / $5,000 / $5,000 (1st–10th).
Efficiency track: $7,000 / $6,000 / $5,000.

Efficiency score (minimize):

```
Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400
```

where `Benchmark` is the `sample_submission.csv` score and `maxAUC` the best private-LB AUC.
Eligible if selected as a final submission and ranked above the sample-submission benchmark.

## Rules / constraints

- **Code competition**: submit via Kaggle Notebooks only. CPU ≤ 9h, GPU ≤ 9h, **internet disabled**.
- Output must be named `submission.csv`. Synchronous rerun scoring.
- 5 submissions/day; 2 final submissions selected; max team size 5.
- Freely & publicly available external data and pre-trained models allowed.
- Identity verification required to submit.
- Winner license: **CC-BY-NC 4.0** (code + winning submission open sourced).
  Data license: RSNA MIRA (http://rsna.org/mira-license), commercial + academic use permitted.
- Extra host asks for winners: short video presenting the approach; publish code + weights link
  on the competition forum; share the final model publicly for open distribution/validation.

## Implications for our approach

- Test-time input is **images only** — the reports are a training-time label source, not a feature.
  Plan: a report → label extractor (multilingual) to pseudo-label the unlabeled majority, then
  train a study-level image model on the expanded label set.
- Series metadata (plane, fluid sensitivity, fat suppression) is available at test time and is the
  natural way to route/select sequences per finding (e.g. fluid-sensitive sagittal for meniscus/ACL).
- 9h GPU cap for ~1300 studies × several series × ~30 slices each makes inference throughput a real
  design constraint — and it is directly scored in the efficiency track.
