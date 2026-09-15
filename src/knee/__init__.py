"""RSNA Knee Abnormality Detection.

Modules are ordered by how much they decide the outcome:

- `reports`  label extraction from the multilingual reports, calibrated against
  the 58 gold studies. The public bar is 0.893 agreement; beating it is the bet.
- `folds`    leak-safe cross-validation (shared reports, shared scanners).
- `fuse`     ensemble fusion fitted on out-of-fold predictions, never the leaderboard.
- `corpus`   study to six anatomical slots, with the empty-slot leak closed.
- `metrics`  macro AUC plus the diagnostics that catch silent failures.
"""

from . import constants, corpus, folds, fuse, metrics, reports

__all__ = ["constants", "corpus", "folds", "fuse", "metrics", "reports"]
