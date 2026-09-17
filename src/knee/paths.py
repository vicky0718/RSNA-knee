"""Where the data is, resolved rather than assumed.

A Kaggle kernel created through the web UI mounts the competition at
`/kaggle/input/<slug>`; one created through the API mounts it at
`/kaggle/input/competitions/<slug>`. Hard-coding either gives a kernel that
runs green, finds nothing, and writes no output — which is how a wasted run
looks from the outside.

`competition_root` checks the candidates and raises with the directory listing
when none of them holds the data.
"""

from __future__ import annotations

import os
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
SENTINEL = "train.csv"


def candidate_roots(competition: str = COMPETITION) -> list[Path]:
    """Every place the competition data is known to mount, plus local overrides."""
    roots = [
        Path("/kaggle/input") / competition,
        Path("/kaggle/input/competitions") / competition,
        Path("data"),
    ]
    override = os.environ.get("KNEE_DATA")
    if override:
        roots.insert(0, Path(override))
    return roots


def competition_root(competition: str = COMPETITION, sentinel: str = SENTINEL) -> Path:
    """The directory holding the competition data.

    Raises rather than returning a guess: a missing mount must stop the run, not
    quietly produce an empty result.
    """
    for root in candidate_roots(competition):
        if (root / sentinel).exists():
            return root
    listing = sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else []
    raise FileNotFoundError(
        f"no {sentinel} under any of {[str(r) for r in candidate_roots(competition)]}; "
        f"/kaggle/input holds {listing}. Set KNEE_DATA to point at the data."
    )
