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

#: (name, plane, fluid_sensitive, fat_suppressed); None means "either".
SLOTS: tuple[tuple[str, str, bool | None, bool], ...] = (
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
)

SLOT_NAMES: tuple[str, ...] = tuple(s[0] for s in SLOTS)

#: Slot indices each finding is normally read from. Anatomy, not learned.
SLOT_PRIOR_TABLE: dict[str, tuple[int, ...]] = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}

SLOT_PRIOR_STRENGTH = 0.55

#: Crop a fixed physical field of view, never a fixed pixel count: a 140 mm crop
#: spans 398-896 px across this dataset's series (a 2.3x range).
CROP_MM = 130.0

#: Observed slot coverage over the 4,407 training studies. Two slots are mostly
#: empty; feeding zeros for them is the leak we fix in corpus.fill_missing_slots.
SLOT_COVERAGE = {
    "SAG_FLUID_FS": 0.942,
    "COR_FLUID_FS": 0.964,
    "AX_FLUID_FS": 1.000,
    "SAG_FLUID_NOFS": 0.968,
    "COR_T1": 0.773,
    "SAG_T1": 0.194,
}

N_TRAIN_STUDIES = 4407
N_GOLD_LABELLED = 58
