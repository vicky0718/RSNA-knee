import numpy as np
import pandas as pd
import pytest

from knee import folds
from knee.constants import TARGETS


def test_normalise_strips_digits_and_accents():
    assert folds.normalise_report("Rotura del LCA, 12 mm.") == "rotura del lca mm"


def test_report_clusters_group_exact_and_near_duplicates():
    base = "the anterior cruciate ligament is intact with no meniscal tear and a small effusion"
    reports = pd.Series([base, base, base + " mild synovitis", "patellar fracture noted here"])
    clusters = folds.report_clusters(reports, threshold=0.6)
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] != clusters[0]


def test_scanner_keys_fall_back_to_study_id_without_headers():
    frame = pd.DataFrame(index=["s1", "s2"], data={"unrelated": [1, 2]})
    assert folds.scanner_keys(frame).nunique() == 2


def test_make_folds_respects_groups():
    rng = np.random.default_rng(0)
    n = 300
    labels = pd.DataFrame((rng.random((n, 12)) < 0.2).astype(float), columns=list(TARGETS))
    scanner = pd.Series(rng.integers(0, 20, n), name="scanner_key")
    clusters = pd.Series([f"{s}_{i // 3}" for i, s in enumerate(scanner)], name="report_cluster")
    groups = folds.combine_groups(clusters, scanner)
    fold = folds.make_folds(labels, groups)
    assert fold.nunique() == 5
    folds.assert_no_leak(fold, groups, clusters, scanner)


def test_assert_no_leak_raises_when_a_group_spans_folds():
    fold = pd.Series(np.arange(10) % 5)
    everything = pd.Series(np.zeros(10, dtype=int), name="everything")
    with pytest.raises(AssertionError, match="span"):
        folds.assert_no_leak(fold, everything)


def test_combine_groups_refuses_to_collapse():
    rng = np.random.default_rng(1)
    n = 300
    scanner = pd.Series(rng.integers(0, 20, n), name="scanner_key")
    bridging = pd.Series(rng.integers(0, 150, n), name="report_cluster")
    with pytest.raises(ValueError, match="collapses"):
        folds.combine_groups(bridging, scanner)
    diagnostics = folds.group_diagnostics(bridging, scanner)
    assert set(diagnostics["key"]) == {"report_cluster", "scanner_key", "union"}


def test_rarest_positive_spreads_scarce_findings():
    labels = pd.DataFrame(0.0, index=range(100), columns=list(TARGETS))
    labels.iloc[:40, 0] = 1.0  # common
    labels.iloc[40:42, 11] = 1.0  # rare
    strata = folds.rarest_positive(labels)
    # the two rare-finding studies share a stratum, distinct from the common one
    assert strata.iloc[40] == strata.iloc[41]
    assert strata.iloc[40] != strata.iloc[0]
    assert (strata == -1).sum() == 58  # studies positive for nothing
