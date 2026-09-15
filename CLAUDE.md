# RSNA Knee — working notes

Target: gold (public cut 0.951). Best public notebook is 0.941 and ~350 teams are on it, so the
job is +0.010 that cannot be forked. Background: `docs/competitive-landscape.md`.
Schedule: `docs/action-plan.md`.

## The three bets

1. **Labels.** 58 of 4,407 studies have gold labels; the rest are supervised by a report in one of
   ~10 languages. **Measured as macro AUC** on those 58 (`bin/score_labels.py`): best public table
   (`llm_labels_v4_blend`) **0.893**, our rule reader 0.748 — a gap of +0.144 [+0.102, +0.188].
   The public table is strong and **is the baseline to build on, not to replace**. Remaining
   headroom is target-specific: Synovitis 0.790, Fracture 0.793, Lateral OA 0.833, Contusion 0.860.
   Everything else it reads at 0.88-0.99.
2. **Honest validation.** The plateau's per-finding fusion weights were fitted by leaderboard
   probes on a 30% split. We fit on OOF and check the gain survives a held-out fold —
   `src/knee/fuse.py::honest_gain`.
3. **The empty-slot leak.** The public pipeline feeds a zero column when a slot is missing
   (Axial no-fat-sat is present for 19.4% of studies — verified against train_series.csv). Its
   author calls it "the cheapest score leak in this whole pipeline" —
   `src/knee/corpus.py::fill_missing_slots`.

## House rules

- **Never fit anything on the public leaderboard.** Weights, thresholds and blend ratios come from
  OOF. The leaderboard is a held-out check, and it is only 30% of the test set.
- **Never trust a CV number without `folds.assert_no_leak`.** Studies share reports and scanners.
- **Reject correlated arms.** Check `metrics.arm_correlation` before adding an ensemble member;
  above ~0.97 it is a 42nd copy of what we already have, whatever it scores alone.
- **Silence is not a negative.** `not_mentioned` gets its own calibrated probability, never 0.
- **Look for ties.** `metrics.tie_report` after every inference run; a block of identical
  predictions means studies are falling through the slot logic.
- **Log every run** in `experiments.csv` before looking at the score.
- **Score label tables by AUC, never by agreement at 0.5.** The public tables hold calibrated
  probabilities; thresholding them throws away what the metric rewards and made a 0.893 table
  look like 0.819. `bin/score_labels.py` leads with AUC for this reason.
- **Quote a number with its interval.** 58 gold studies is a small stick: our reader trails the
  best public table by 0.144 macro AUC, 95% CI [0.102, 0.188].
- **Verify claims against the data — including our own.** The series flags and the slot scheme
  turned out different from their description, and one of our own "corrections" to a published
  figure was itself wrong. `bin/audit_data.py` and `bin/score_labels.py` exist so the next claim
  gets checked, by whoever makes it.

## Loop

```bash
python3 -m pytest tests/ -q          # must stay green; it is the only thing running locally
KAGGLE_USERNAME=you bin/push_kaggle.sh "what changed"
# then in the Kaggle notebook:
#   import sys; sys.path.insert(0, "/kaggle/input/knee-src")
```

Kaggle notebooks have no internet. Anything the submission needs — weights, wheels, DICOM codecs
(`pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `python-gdcm`) — must be attached as a
dataset. A missing codec does not raise; it silently fails to decode a transfer syntax.

Submission budget is 5/day, 2 final: two candidates, one fixed control to detect drift, two held.
