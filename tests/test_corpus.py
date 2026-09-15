import numpy as np
import pandas as pd
import pytest

from knee import corpus


def _series(**overrides):
    rows = [
        dict(SeriesInstanceUID="a", Anatomical_Plane="Sagittal", Fluid_Sensitive=1, Fat_Suppression=1, n_slices=30),
        dict(SeriesInstanceUID="b", Anatomical_Plane="Axial", Fluid_Sensitive=1, Fat_Suppression=1, n_slices=24),
        dict(SeriesInstanceUID="c", Anatomical_Plane="Coronal", Fluid_Sensitive=1, Fat_Suppression=1, n_slices=40),
    ]
    rows.extend(overrides.get("extra", []))
    return pd.DataFrame(rows)


def test_slice_order_follows_geometry_not_input_order():
    iop = np.array([1, 0, 0, 0, 1, 0], float)
    positions = np.array([[0, 0, z] for z in [4, 0, 2, 6, 8]], float)
    ordered = positions[corpus.slice_order(positions, iop)][:, 2]
    assert np.all(np.diff(ordered) > 0)


def test_degenerate_orientation_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        corpus.slice_normal(np.array([1, 0, 0, 1, 0, 0], float))


def test_mm_crop_scales_with_pixel_spacing():
    """Same anatomy, different pixel counts — the reason we crop in millimetres."""
    fine = corpus.mm_crop_box(512, 512, (0.3, 0.3))
    coarse = corpus.mm_crop_box(512, 512, (0.6, 0.6))
    assert (fine[1] - fine[0]) > (coarse[1] - coarse[0])


def test_mm_crop_clips_to_image_bounds():
    r0, r1, c0, c1 = corpus.mm_crop_box(64, 64, (2.0, 2.0))
    assert r0 >= 0 and c0 >= 0 and r1 <= 64 and c1 <= 64


def test_laterality_flip_axis_depends_on_plane():
    image = np.arange(12).reshape(1, 3, 4)
    assert corpus.normalise_laterality(image, "Coronal", "R").tolist() == image.tolist()
    assert not np.array_equal(corpus.normalise_laterality(image, "Coronal", "L"), image)
    coronal = corpus.normalise_laterality(image, "Coronal", "L")
    sagittal = corpus.normalise_laterality(image, "Sagittal", "L")
    assert not np.array_equal(coronal, sagittal)


def test_pick_slots_requires_the_series_descriptors():
    with pytest.raises(KeyError, match="Fluid_Sensitive"):
        corpus.pick_slots(pd.DataFrame({"SeriesInstanceUID": ["a"], "Anatomical_Plane": ["Axial"], "Fat_Suppression": [1]}))


def test_pick_slots_prefers_more_slices():
    extra = [dict(SeriesInstanceUID="a2", Anatomical_Plane="Sagittal", Fluid_Sensitive=1, Fat_Suppression=1, n_slices=45)]
    chosen = corpus.pick_slots(_series(extra=extra))
    assert chosen["SAG_FLUID_FS"] == "a2"


def test_missing_slots_are_substituted_not_zeroed():
    """The public pipeline feeds zeros here; a zero column ties studies together."""
    chosen = corpus.pick_slots(_series())
    fill = corpus.fill_missing_slots(chosen)
    assert len(fill.series) == 6
    assert fill.substituted_from["SAG_T1"].startswith("SAG")  # same plane preferred
    assert fill.real["SAG_T1"] is False
    assert fill.n_real == 3 and fill.usable


def test_study_with_no_series_is_flagged_unusable():
    fill = corpus.fill_missing_slots({})
    assert not fill.usable
    assert fill.series == {}


def test_coverage_report_counts_unusable_studies():
    chosen = corpus.pick_slots(_series())
    table = corpus.coverage_report({"s1": chosen, "s2": {"AX_FLUID_FS": "b"}, "s3": {}})
    assert (table["coverage_filled"] >= table["coverage_raw"]).all()
    assert table.attrs["unusable_studies"] == 1
