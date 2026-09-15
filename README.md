# RSNA Knee Abnormality Detection

Twelve findings per knee MRI study, scored by macro ROC-AUC.
Final submission **2026-10-22**.

- `docs/competition-overview.md` — rules, metric, data layout, timeline
- `docs/competitive-landscape.md` — leaderboard, what the field is running, where its weaknesses are
- `docs/action-plan.md` — dated plan and the decision gate
- `CLAUDE.md` — house rules for anyone (human or agent) working in here

## Layout

| path | what |
|---|---|
| `src/knee/reports.py` | report → calibrated soft labels, scored against the 58 gold studies |
| `src/knee/folds.py` | leak-safe CV: shared-report and shared-scanner grouping |
| `src/knee/fuse.py` | rank-space fusion fitted on OOF, with a held-out check on per-target routing |
| `src/knee/corpus.py` | study → six anatomical slots, empty slots substituted rather than zeroed |
| `src/knee/metrics.py` | macro AUC, per-target headroom, tie and arm-correlation diagnostics |
| `bin/push_kaggle.sh` | ship `src/` to Kaggle as a utility dataset |
| `experiments.csv` | one row per run: config, OOF, LB, runtime |

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -q
```
