import os

import pytest

from knee import paths


def test_override_wins(tmp_path, monkeypatch):
    (tmp_path / "train.csv").write_text("StudyInstanceUID\n")
    monkeypatch.setenv("KNEE_DATA", str(tmp_path))
    assert paths.competition_root() == tmp_path


def test_missing_data_raises_with_the_listing(monkeypatch, tmp_path):
    """A missing mount must stop the run; a kernel that skips looks like success."""
    monkeypatch.setenv("KNEE_DATA", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no train.csv"):
        paths.competition_root()


def test_both_kaggle_mount_points_are_candidates():
    roots = [str(r) for r in paths.candidate_roots()]
    assert "/kaggle/input/rsna-knee-abnormality-detection" in roots
    assert "/kaggle/input/competitions/rsna-knee-abnormality-detection" in roots
