"""Competition constants and the public stack's slot scheme.

The twelve targets and their order are fixed by the submission format. The slot
scheme, slot priors and mm-crop are the public 0.941 pipeline's conventions,
recorded here so our code and any public checkpoint agree on layout.
See docs/competitive-landscape.md for provenance.
"""

from __future__ import annotations

TARGETS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

ID_COLUMN = "StudyInstanceUID"
SUBMISSION_FILENAME = "submission.csv"

#: (name, plane, fat_suppressed).
#:
#: Slots key on plane and fat-suppression only. `Fluid_Sensitive` and
#: `Fat_Suppression` are *identical for all 24,371 training series* — verified,
#: not assumed — so adding fluid-sensitivity to the rule buys nothing and can
#: define a slot that no series can ever fill. The data description warns the two
#: flags "are not necessarily equivalent for every case", so the test set may yet
#: separate them; keying on fat-suppression keeps every slot fillable either way.
SLOTS: tuple[tuple[str, str, bool], ...] = (
    ("SAG_FS", "Sagittal", True),
    ("COR_FS", "Coronal", True),
    ("AX_FS", "Axial", True),
    ("SAG_NOFS", "Sagittal", False),
    ("COR_NOFS", "Coronal", False),
    ("AX_NOFS", "Axial", False),
)

SLOT_NAMES: tuple[str, ...] = tuple(s[0] for s in SLOTS)

#: Slots each finding is normally read from. Anatomy, not learned.
#:
#: Named rather than indexed on purpose. The public pipeline stores these as
#: positions into its own slot list, whose entries 3 and 5 ("SAG_FLUID_NOFS" and
#: "SAG_T1") both resolve to the same sagittal non-fat-sat series once selection
#: keys on fat-suppression — so it carries a duplicated slot and no axial
#: non-fat-sat slot at all. Our list is the six distinct plane x fat-suppression
#: combinations; these priors are its scheme translated across, which is why
#: AX_NOFS appears nowhere: nothing in the public table maps to it.
SLOT_PRIOR_TABLE: dict[str, tuple[str, ...]] = {
    "ACL": ("SAG_FS", "SAG_NOFS"),
    "MCL": ("COR_FS", "COR_NOFS"),
    "Medial Meniscus": ("SAG_FS", "COR_FS", "SAG_NOFS", "COR_NOFS"),
    "Lateral Meniscus": ("SAG_FS", "COR_FS", "SAG_NOFS", "COR_NOFS"),
    "Medial OA": ("COR_FS", "COR_NOFS", "SAG_NOFS"),
    "Lateral OA": ("COR_FS", "COR_NOFS", "SAG_NOFS"),
    "PF OA": ("SAG_FS", "AX_FS", "SAG_NOFS"),
    "Effusion": ("SAG_FS", "AX_FS"),
    "Synovitis": ("SAG_FS", "AX_FS"),
    "Baker's": ("SAG_FS",),
    "Contusion": ("SAG_FS", "COR_FS", "AX_FS"),
    "Fracture": ("SAG_FS", "COR_FS", "AX_FS", "COR_NOFS", "SAG_NOFS"),
}


def slot_prior_indices(target: str) -> tuple[int, ...]:
    """Slot positions for a finding, for building a prior over the slot axis."""
    return tuple(SLOT_NAMES.index(name) for name in SLOT_PRIOR_TABLE[target])


SLOT_PRIOR_STRENGTH = 0.55

#: Crop a fixed physical field of view, never a fixed pixel count: a 140 mm crop
#: spans 398-896 px across this dataset's series (a 2.3x range).
CROP_MM = 130.0

#: Observed slot coverage over the 4,407 training studies. Two slots are mostly
#: empty; feeding zeros for them is the leak we fix in corpus.fill_missing_slots.
#: Measured over the 4,407 training studies (bin/audit_data.py), matching the
#: public pipeline's own figures to three decimals.
SLOT_COVERAGE = {
    "SAG_FS": 0.942,
    "COR_FS": 0.964,
    "AX_FS": 1.000,
    "SAG_NOFS": 0.968,
    "COR_NOFS": 0.773,
    "AX_NOFS": 0.194,
}

N_TRAIN_STUDIES = 4407
N_GOLD_LABELLED = 58
