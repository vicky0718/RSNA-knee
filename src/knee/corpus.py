"""Study -> six anatomical slots, with the empty-slot leak closed.

Slice order comes from geometry, not filenames; the crop is a physical field of
view, not a pixel count (a 140 mm crop spans 398-896 px across this dataset, a
2.3x range, so a fixed pixel crop hands the model a different anatomy per
scanner); and laterality is normalised so "medial" means one side of the image.

The part that is ours rather than inherited is `fill_missing_slots`. Slot
coverage over the training set is uneven — Axial no-fat-sat is present for 19.4%
of studies and Coronal no-fat-sat for 77.3% — and the public pipeline feeds a
column of zeros for the rest. Its own author calls that "the cheapest score leak
in this whole pipeline": a study with no usable slot can only be given a
constant, and under ROC-AUC a block of identical predictions earns half credit
against everything it ties with. We substitute the most similar available slot
and record what was substituted, so a study never becomes a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import SLOTS, CROP_MM


def slice_normal(orientation: np.ndarray) -> np.ndarray:
    """Slice normal from ImageOrientationPatient's two row/column direction cosines."""
    orientation = np.asarray(orientation, dtype=float).reshape(6)
    normal = np.cross(orientation[:3], orientation[3:])
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("degenerate ImageOrientationPatient: row and column are parallel")
    return normal / norm


def slice_order(positions: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Indices that sort slices along the acquisition axis.

    Sorting by filename or by InstanceNumber is what produces silently scrambled
    volumes: neither is guaranteed to follow the anatomy.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    projection = positions @ slice_normal(orientation)
    return np.argsort(projection, kind="mergesort")


def mm_crop_box(
    rows: int, cols: int, pixel_spacing: tuple[float, float], mm: float = CROP_MM
) -> tuple[int, int, int, int]:
    """Centred crop covering `mm` millimetres, clipped to the image."""
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    if row_mm <= 0 or col_mm <= 0:
        raise ValueError(f"invalid PixelSpacing {pixel_spacing!r}")
    half_r = min(int(round(mm / row_mm / 2)), rows // 2)
    half_c = min(int(round(mm / col_mm / 2)), cols // 2)
    cr, cc = rows // 2, cols // 2
    return cr - half_r, cr + half_r, cc - half_c, cc + half_c


def normalise_laterality(image: np.ndarray, plane: str, side: str | None) -> np.ndarray:
    """Flip left knees to match right ones so medial and lateral are consistent.

    Coronal and axial acquisitions mirror left-right; sagittal images mirror
    front-back, which is why the flip axis depends on the plane.
    """
    if side is None or str(side).upper().startswith("R"):
        return image
    return np.flip(image, axis=-1) if plane in ("Coronal", "Axial") else np.flip(image, axis=-2)


def pick_slots(series: pd.DataFrame) -> dict[str, str]:
    """Choose one series per anatomical slot for a single study.

    Expects columns SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive,
    Fat_Suppression and n_slices. Ties break on slice count: more slices means
    more of the joint covered.
    """
    required = {"SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"}
    missing = required - set(series.columns)
    if missing:
        raise KeyError(f"series frame is missing {sorted(missing)}")
    frame = series.copy()
    if "n_slices" not in frame.columns:
        frame["n_slices"] = 0
    chosen: dict[str, str] = {}
    for name, plane, fluid, fat in SLOTS:
        hit = (frame["Anatomical_Plane"] == plane) & (frame["Fat_Suppression"].astype(int) == int(fat))
        if fluid is not None:
            hit &= frame["Fluid_Sensitive"].astype(int) == int(fluid)
        candidates = frame[hit]
        if len(candidates):
            best = candidates.sort_values("n_slices", ascending=False).iloc[0]
            chosen[name] = str(best["SeriesInstanceUID"])
    return chosen


def _similarity(a: tuple, b: tuple) -> int:
    """How interchangeable two slots are: same plane counts most, then contrast."""
    _, plane_a, fluid_a, fat_a = a
    _, plane_b, fluid_b, fat_b = b
    return 4 * (plane_a == plane_b) + 2 * (fluid_a == fluid_b) + 1 * (fat_a == fat_b)


def substitution_order(slot_name: str) -> list[str]:
    """Slots to fall back to, most similar first."""
    target = next(s for s in SLOTS if s[0] == slot_name)
    others = [s for s in SLOTS if s[0] != slot_name]
    return [s[0] for s in sorted(others, key=lambda s: -_similarity(target, s))]


@dataclass(frozen=True)
class SlotFill:
    """The filled slot map plus an honest record of what is real."""

    series: dict[str, str]
    real: dict[str, bool]
    substituted_from: dict[str, str]

    @property
    def n_real(self) -> int:
        return sum(self.real.values())

    @property
    def usable(self) -> bool:
        """False only when a study has no series at all — the constant-prediction case."""
        return self.n_real > 0


def fill_missing_slots(chosen: dict[str, str]) -> SlotFill:
    """Fill empty slots from the most similar available one instead of with zeros.

    A substituted slot is still worse than a real one, which is what `real` is
    for: pass it to the model as a mask so it can discount the copy rather than
    treat it as an independent view.
    """
    series = dict(chosen)
    real = {name: name in chosen for name, *_ in SLOTS}
    substituted: dict[str, str] = {}
    if not chosen:
        return SlotFill(series, real, substituted)
    for name, *_ in SLOTS:
        if name in series:
            continue
        for donor in substitution_order(name):
            if donor in chosen:
                series[name] = chosen[donor]
                substituted[name] = donor
                break
    return SlotFill(series, real, substituted)


def coverage_report(slot_maps: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Per-slot coverage before and after substitution, plus the unusable studies.

    Run this on the training set before training anything. The last column is the
    number of studies that could only ever be given a constant prediction.
    """
    rows = []
    total = max(len(slot_maps), 1)
    filled = {study: fill_missing_slots(chosen) for study, chosen in slot_maps.items()}
    for name, *_ in SLOTS:
        raw = sum(1 for chosen in slot_maps.values() if name in chosen)
        after = sum(1 for f in filled.values() if name in f.series)
        rows.append(
            {
                "slot": name,
                "coverage_raw": raw / total,
                "coverage_filled": after / total,
                "substituted": (after - raw) / total,
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["unusable_studies"] = sum(1 for f in filled.values() if not f.usable)
    return table
