import numpy as np
import pytest

from knee import fuse
from knee.metrics import macro_auc


def _arms(rng, n=1200, t=12, split=6):
    y = (rng.random((n, t)) < 0.25).astype(float)
    signal = y * 2 - 1
    strong_first = np.where(np.arange(t) < split, 1.4, 0.35)
    a = signal * strong_first + rng.normal(0, 1, (n, t))
    b = signal * strong_first[::-1] + rng.normal(0, 1, (n, t))
    return y, {"A": a, "B": b}


def test_rank_normalise_is_monotone_and_bounded():
    x = np.array([[3.0], [1.0], [2.0]])
    r = fuse.rank_normalise(x)
    assert r.min() == 0.0 and r.max() == 1.0
    assert np.argsort(r[:, 0]).tolist() == np.argsort(x[:, 0]).tolist()


def test_blend_weights_must_match_arm_count():
    arms = {"A": np.zeros((5, 12)), "B": np.zeros((5, 12))}
    with pytest.raises(ValueError, match="3 arms|expected weights"):
        fuse.blend(arms, np.ones((3, 12)) / 3)


def test_routing_beats_flat_when_arms_specialise():
    y, arms = _arms(np.random.default_rng(7))
    weights = fuse.fit_weights(arms, y)
    flat = macro_auc(y, fuse.blend(arms, np.full((2, y.shape[1]), 0.5)))
    routed = macro_auc(y, fuse.blend(arms, weights.to_numpy()))
    assert routed > flat
    assert weights.loc["A", "ACL"] > weights.loc["B", "ACL"]


def test_honest_gain_calls_real_routing_real():
    rng = np.random.default_rng(7)
    y, arms = _arms(rng)
    fold = rng.integers(0, 5, len(y))
    gain = fuse.honest_gain(arms, y, fold)
    assert gain.real
    assert gain.out_of_sample > gain.global_baseline


def test_honest_gain_calls_noise_routing_noise():
    """Two interchangeable arms: per-target weights must not survive a held-out fold.

    This is the probe-22 failure mode — twelve correlated columns fitted to one
    split — and the reason we never ship a per-finding weight untested.
    """
    rng = np.random.default_rng(11)
    n, t = 1200, 12
    y = (rng.random((n, t)) < 0.25).astype(float)
    signal = y * 2 - 1
    arms = {
        "C": signal * 0.8 + rng.normal(0, 1, (n, t)),
        "D": signal * 0.8 + rng.normal(0, 1, (n, t)),
    }
    gain = fuse.honest_gain(arms, y, rng.integers(0, 5, n))
    assert gain.in_sample > gain.out_of_sample  # fitting the split flatters itself
    assert not gain.real


def test_shrink_to_global_interpolates():
    y, arms = _arms(np.random.default_rng(3))
    per_target = fuse.fit_weights(arms, y)
    global_w = fuse.fit_global_weights(arms, y)
    assert fuse.shrink_to_global(per_target, global_w, 0.0).equals(per_target)
    full = fuse.shrink_to_global(per_target, global_w, 1.0)
    assert np.allclose(full.to_numpy(), global_w.to_numpy()[:, None])
    with pytest.raises(ValueError):
        fuse.shrink_to_global(per_target, global_w, 1.5)
