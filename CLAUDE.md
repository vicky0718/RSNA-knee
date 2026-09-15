# RSNA Knee — working notes

Target: gold (public cut 0.951). Best public notebook is 0.941 and ~350 teams are on it, so the
job is +0.010 that cannot be forked. Background: `docs/competitive-landscape.md`.
Schedule: `docs/action-plan.md`.

## The three bets

1. **Labels.** 58 of 4,407 studies have gold labels; the rest are supervised by a report in one of
   ~10 languages. **Measured** on those 58 (`bin/score_labels.py`): best public table **0.819**,
   our rule reader **0.777**, all-negative baseline 0.655. The published "0.893, regex 0.814" does
   not reproduce. Beat 0.819 — `src/knee/rules.py`, `src/knee/reports.py`.
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
- **Quote a number with its interval.** 58 gold studies is a small stick: the gap between our
  reader and the best public table is +0.042 with a 95% CI of [+0.010, +0.079].
- **Verify claims against the data.** The published label-agreement figure, the series flags and
  the slot scheme each turned out different from their description; `bin/audit_data.py` and
  `bin/score_labels.py` exist so the next claim gets checked too.

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
