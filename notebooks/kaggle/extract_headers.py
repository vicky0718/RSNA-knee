"""Extract one DICOM header per series, to recover scanner identity.

Scanner and site identity is the leak we have not been able to guard: it lives
only in the DICOM headers, and the headers are only on Kaggle. This reads one
file per series -- headers only, never pixels -- and writes the per-series table
the fold builder needs.

It also times itself, because the same walk is the front half of the submission
path and we need to know what ~1,322 test studies will cost inside the 9-hour
budget before the last week.

Standalone by design: this is a one-off extraction with no inference twin, so
there is no pipeline copy for it to drift from.
"""

import os
import time
from pathlib import Path

import pandas as pd
import pydicom

COMPETITION = "rsna-knee-abnormality-detection"

# The mount point depends on how the kernel was created: the web UI attaches the
# competition at /kaggle/input/<slug>, the API at /kaggle/input/competitions/<slug>.
# Resolve both, and fail loudly if neither is there -- a missing mount that
# "skips" is a green run that did nothing.
CANDIDATES = [
    Path("/kaggle/input") / COMPETITION,
    Path("/kaggle/input/competitions") / COMPETITION,
]
ROOT = next((c for c in CANDIDATES if (c / "train.csv").exists()), None)
if ROOT is None:
    listing = sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else []
    raise SystemExit(f"competition data not mounted; /kaggle/input holds {listing}")
print(f"competition data at {ROOT}")
OUT = Path("/kaggle/working")

# Read only what identifies the machine, the geometry and the side. Anything
# else is pixels we are not paying for.
TAGS = [
    "Manufacturer",
    "ManufacturerModelName",
    "StationName",
    "DeviceSerialNumber",
    "SoftwareVersions",
    "MagneticFieldStrength",
    "InstitutionName",
    "Laterality",
    "ImageLaterality",
    "BodyPartExamined",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "ScanOptions",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "RescaleSlope",
    "RescaleIntercept",
    "ImageOrientationPatient",
    "PatientSex",
    "PatientAge",
]


def read_series(series_dir):
    """One header from one file, plus how many files the series holds."""
    files = sorted(os.listdir(series_dir))
    if not files:
        return None
    path = os.path.join(series_dir, files[len(files) // 2])  # a middle slice is the most typical
    try:
        header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as error:  # a series we cannot read is a fact, not a crash
        return {"n_slices": len(files), "read_error": type(error).__name__}
    row = {"n_slices": len(files), "read_error": ""}
    for tag in TAGS:
        value = getattr(header, tag, None)
        if value is None:
            continue
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            value = "|".join(str(v) for v in value)
        row[tag] = str(value)
    return row


def walk(split):
    base = ROOT / f"{split}_series"
    if not base.exists():
        if split == "train":
            raise SystemExit(f"{base} is missing: nothing to extract")
        print(f"{base} absent; skipping")  # test_series is legitimately absent at times
        return pd.DataFrame()
    rows = []
    start = time.time()
    studies = sorted(os.listdir(base))
    for i, study in enumerate(studies, 1):
        study_dir = base / study
        for series in sorted(os.listdir(study_dir)):
            row = read_series(study_dir / series)
            if row is None:
                continue
            row["StudyInstanceUID"] = study
            row["SeriesInstanceUID"] = series
            rows.append(row)
        if i % 250 == 0 or i == len(studies):
            rate = i / max(time.time() - start, 1e-6)
            print(
                f"  {split}: {i}/{len(studies)} studies, {len(rows)} series, "
                f"{rate:.1f} studies/s, eta {(len(studies) - i) / max(rate, 1e-9) / 60:.1f} min",
                flush=True,
            )
    elapsed = time.time() - start
    print(f"{split}: {len(rows)} series in {elapsed / 60:.1f} min ({len(studies) / elapsed:.1f} studies/s)")
    print(f"  -> ~1322 test studies would take {1322 / max(len(studies) / elapsed, 1e-9) / 60:.1f} min at this rate")
    return pd.DataFrame(rows)


def scanner_key(frame):
    """One key per study, from whatever identifies the machine.

    Falls back to the study id when nothing does, which makes that study its own
    group -- the safe direction to be wrong in for a fold guard.
    """
    columns = [
        c
        for c in ("Manufacturer", "ManufacturerModelName", "StationName", "DeviceSerialNumber",
                  "MagneticFieldStrength", "InstitutionName")
        if c in frame.columns
    ]
    print(f"scanner key built from: {columns}")
    if not columns:
        keys = frame.groupby("StudyInstanceUID").size().index.to_series()
        return keys.rename("scanner_key").reset_index(drop=False)
    per_series = frame[columns].fillna("").astype(str).agg("|".join, axis=1)
    per_study = (
        pd.DataFrame({"StudyInstanceUID": frame["StudyInstanceUID"], "key": per_series})
        .groupby("StudyInstanceUID")["key"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    )
    per_study = per_study.replace("", pd.NA)
    per_study = per_study.fillna(pd.Series(per_study.index, index=per_study.index))
    return per_study.rename("scanner_key").reset_index()


train = walk("train")
if not train.empty:
    train.to_csv(OUT / "train_series_headers.csv", index=False)
    keys = scanner_key(train)
    keys.to_csv(OUT / "scanner_keys.csv", index=False)
    print(f"\n{keys['scanner_key'].nunique()} distinct scanner keys over {len(keys)} studies")
    print(keys["scanner_key"].value_counts().head(8).to_string())
    print(f"\nlargest scanner group holds {keys['scanner_key'].value_counts().iloc[0] / len(keys):.1%} of studies")
    print("\ntag availability (share of series carrying each):")
    print((train.notna().mean().sort_values(ascending=False) * 100).round(1).to_string())
    if "read_error" in train.columns:
        errors = train["read_error"].replace("", pd.NA).dropna()
        print(f"\nunreadable series: {len(errors)}")

test = walk("test")
if not test.empty:
    test.to_csv(OUT / "test_series_headers.csv", index=False)
print("\ndone")
