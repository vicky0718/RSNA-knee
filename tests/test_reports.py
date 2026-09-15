import numpy as np
import pandas as pd

from knee import reports
from knee.constants import TARGETS
from knee.reports import LabelState


def _states(rng, n=400, silence=0.3):
    index = [f"s{i}" for i in range(n)]
    truth = pd.DataFrame(
        (rng.random((n, 12)) < 0.25).astype(float), index=index, columns=list(TARGETS)
    )
    raw = np.where(truth.to_numpy() > 0.5, LabelState.POSITIVE.value, LabelState.NEGATIVE.value)
    raw = np.where(rng.random((n, 12)) < silence, LabelState.UNMENTIONED.value, raw)
    return truth, pd.DataFrame(raw, index=index, columns=list(TARGETS))


def test_calibration_keeps_positive_above_unmentioned_above_negative():
    truth, states = _states(np.random.default_rng(3))
    table = reports.calibrate(states, truth.iloc[:58]).table
    assert (table["positive"] > table["not_mentioned"]).all()
    assert (table["not_mentioned"] > table["negative"]).all()


def test_unmentioned_never_collapses_to_zero():
    """Silence is not a negative. A zero here teaches absence of evidence as evidence."""
    truth, states = _states(np.random.default_rng(5))
    states.iloc[:, :] = LabelState.UNMENTIONED.value  # extractor read nothing at all
    table = reports.calibrate(states, truth.iloc[:58]).table
    assert (table["not_mentioned"] > 0).all()


def test_soft_targets_stay_probabilities():
    truth, states = _states(np.random.default_rng(3))
    probs = reports.calibrate(states, truth.iloc[:58]).apply(states)
    assert probs.shape == states.shape
    assert probs.min() > 0.0 and probs.max() < 1.0


def test_confidence_weights_discount_silence():
    _, states = _states(np.random.default_rng(3))
    weights = reports.confidence_weights(states)
    silent = states.to_numpy() == LabelState.UNMENTIONED.value
    assert weights[silent].max() < weights[~silent].min()


def test_agreement_reports_macro_over_the_gold_studies():
    truth, states = _states(np.random.default_rng(3))
    table = reports.agreement_vs_gold(states, truth.iloc[:58])
    macro = table[table["target"] == "MACRO"].iloc[0]
    assert 0.0 <= macro["agreement"] <= 1.0
    assert macro["n"] == 58


def test_parse_extraction_survives_fences_and_junk():
    raw = '```json\n{"ACL": {"state": "positive", "confidence": 0.9, "evidence": "rotura del LCA"},'
    raw += ' "MCL": {"state": "bogus"}}\n```'
    parsed = reports.parse_extraction(raw)
    assert parsed["ACL"]["state"] == LabelState.POSITIVE.value
    assert parsed["ACL"]["evidence"] == "rotura del LCA"
    assert parsed["MCL"]["state"] == LabelState.UNMENTIONED.value
    assert reports.parse_extraction("the model refused")["ACL"]["confidence"] == 0.0


def test_disagreements_lists_contested_cells():
    truth, _ = _states(np.random.default_rng(3))
    other = truth.copy()
    other.iloc[0, 0] = 1 - other.iloc[0, 0]
    contested = reports.disagreements({"a": truth, "b": other})
    assert len(contested) == 1
    assert contested.iloc[0]["target"] == TARGETS[0]
