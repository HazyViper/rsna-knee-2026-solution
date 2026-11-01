"""Tests for src/eval/metrics.py.

Runs under pytest, and standalone:  python tests/test_metrics.py

Three groups:
  1. synthetic fixtures whose answer is known analytically, including the
     all-tied-prediction case that separates Somers' D from Goodman-Kruskal's
     gamma -- that one is the check on the denominator fix;
  2. an equivalence check: with a single site, within_site_c must reproduce
     somers_d exactly;
  3. paired_58 / verdict_58 on synthetic results.

The research repository also locks the metric to six decimals on its shipped
out-of-fold predictions and re-derives two published 58-study accountings.
Those tests read study-level files derived from the competition data, which
cannot be published, so this copy leaves them out.
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import (Paired58Result, paired_58,  # noqa: E402
                              paired_bootstrap, somers_d, verdict_58,
                              within_site_c)
from src.eval.metrics import _MDD_Z, _pair_counts          # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

SIX_DP = 5e-7          # |a - b| < 5e-7  <=>  a and b agree to six decimals


def _gamma(y, s) -> float:
    """Goodman-Kruskal's gamma, C / (C + D), for contrast only.  Not exported
    from the module; written out here so the tests can show what the corrected
    denominator changes."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    c, d, _ = _pair_counts(y[m], s[m])
    return (c - d) / (c + d) if (c + d) else float("nan")


def _auc_rank(y, s) -> float:
    """ROC-AUC via the rank statistic, ties at half credit.  Independent of the
    module under test, so it can referee it."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


# ===========================================================================
# 1. synthetic fixtures with an analytic answer
# ===========================================================================
def test_perfect_agreement_is_one():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert somers_d(y, y) == 1.0
    assert somers_d(y, 3.0 * y + 7.0) == 1.0          # any increasing map
    assert somers_d(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0


def test_perfect_inversion_is_minus_one():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert somers_d(y, -y) == -1.0
    assert somers_d(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1])) == -1.0


def test_all_tied_predictions_is_zero_where_gamma_is_undefined():
    """The denominator fix, stated as a test.

    Every pair is a tied prediction.  Somers' D counts those pairs and scores
    the model 0 -- it discriminates nothing, which is true and measurable.
    Gamma drops them, its denominator becomes empty, and it returns nan."""
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    s = np.full(5, 0.5)
    assert somers_d(y, s) == 0.0
    assert np.isnan(_gamma(y, s))

    # and the constant model is 0 for a continuous target too
    rng = np.random.default_rng(0)
    yc = rng.random(200)
    assert somers_d(yc, np.full(200, 0.3)) == 0.0
    assert np.isnan(_gamma(yc, np.full(200, 0.3)))


def test_partial_ties_inflate_gamma_but_not_somers_d():
    """Gamma is graded only on the pairs the model has an opinion about, so a
    coarse informative model is flattered; Somers' D is not.  This is why the
    two must not be swapped for one another."""
    rng = np.random.default_rng(11)
    y = (rng.random(400) > 0.5).astype(float)
    s = np.round(np.clip(0.35 * y + 0.65 * rng.random(400), 0, 1))   # 2 levels
    d, g = somers_d(y, s), _gamma(y, s)
    assert g > d + 0.05, (g, d)


def test_binary_target_is_roc_auc():
    """(D + 1) / 2 is exactly ROC-AUC, ties and all, against an independent
    rank-based implementation."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(20, 300))
        y = (rng.random(n) > 0.6).astype(float)
        s = np.round(rng.random(n), int(rng.integers(1, 4)))   # forces ties
        a = _auc_rank(y, s)
        if not np.isfinite(a):
            continue
        assert abs((somers_d(y, s) + 1) / 2 - a) < 1e-12


def test_denominator_does_not_depend_on_the_model():
    """Same target, same rows, three very different score vectors -> the same
    pair count.  Gamma cannot say this."""
    rng = np.random.default_rng(5)
    y = rng.random(300)
    site = rng.integers(0, 7, 300)
    dens = set()
    for s in (rng.random(300), np.round(rng.random(300), 1), np.full(300, 0.3)):
        r = within_site_c(y, s, site)
        dens.add(r["per_label"]["0"]["n_pairs"])
    assert len(dens) == 1, dens


def test_degenerate_inputs_return_nan_not_exceptions():
    ok = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.isnan(somers_d(np.array([1.0, 1.0, 1.0]), np.array([1., 2., 3.])))
    assert np.isnan(somers_d(np.ones(4), ok))              # one class only
    assert np.isnan(somers_d(np.zeros(4), ok))
    assert np.isnan(somers_d(np.array([0.0]), np.array([1.0])))     # n < 2
    assert np.isnan(somers_d(np.array([]), np.array([])))
    assert np.isnan(somers_d(np.full(5, np.nan), np.arange(5.)))
    assert np.isnan(somers_d(ok, np.full(4, np.nan)))
    assert np.isnan(somers_d(np.array([0., 1., np.inf]),
                             np.array([np.nan, 1., 2.])))


def test_nan_rows_are_dropped_pairwise_not_propagated():
    y = np.array([0.0, 1.0, 0.0, 1.0, np.nan])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
    assert somers_d(y, s) == 1.0
    y2 = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    s2 = np.array([0.1, 0.9, 0.2, 0.8, np.nan])
    assert somers_d(y2, s2) == 1.0


def test_within_site_c_reports_nan_labels_and_excludes_them_from_macro():
    y = np.column_stack([np.array([0., 1., 0., 1.]), np.ones(4)])
    s = np.column_stack([np.array([.1, .9, .2, .8]), np.array([.1, .2, .3, .4])])
    r = within_site_c(y, s, np.zeros(4, int))
    assert r["macro_c"] == 1.0                  # label 1 is undefined, excluded
    assert r["undefined_labels"] == ["1"]
    assert r["n_labels_used"] == 1
    assert r["n"] == 4


def test_within_site_c_pools_then_divides():
    """Pooling is not the same as averaging the per-site values, and a small
    site must not weigh as much as a large one."""
    rng = np.random.default_rng(7)
    y, s = rng.random(400), rng.random(400)
    site = np.r_[np.zeros(390, int), np.ones(10, int)]
    pooled = within_site_c(y, s, site)["macro_c"]
    each = [within_site_c(y[site == k], s[site == k],
                          np.zeros(int((site == k).sum()), int))["macro_c"]
            for k in (0, 1)]
    assert abs(pooled - np.mean(each)) > 1e-6
    big = within_site_c(y[site == 0], s[site == 0], np.zeros(390, int))["macro_c"]
    assert abs(pooled - big) < abs(pooled - each[1])


def test_fold_selection_is_explicit_and_has_no_default():
    rng = np.random.default_rng(9)
    n = 200
    y, s = rng.random(n), rng.random(n)
    site = rng.integers(0, 3, n)
    folds = rng.integers(0, 5, n)
    assert within_site_c(y, s, site)["n"] == n
    assert within_site_c(y, s, site, folds=folds)["n"] == n        # all folds
    r0 = within_site_c(y, s, site, folds=folds, fold_mask=[0])
    assert r0["n"] == int((folds == 0).sum())
    r03 = within_site_c(y, s, site, folds=folds, fold_mask=[0, 3])
    assert r03["n"] == int(np.isin(folds, [0, 3]).sum())
    rb = within_site_c(y, s, site, fold_mask=(folds == 0))
    assert rb["n"] == r0["n"] and rb["macro_c"] == r0["macro_c"]
    for bad in ([0], np.arange(n)):
        try:
            within_site_c(y, s, site, fold_mask=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-boolean fold_mask without folds must raise")


# ===========================================================================
# 2. one site => within_site_c is the global somers_d
# ===========================================================================
def test_single_site_matches_somers_d():
    rng = np.random.default_rng(13)
    for _ in range(20):
        n = int(rng.integers(30, 300))
        y = np.round(rng.random(n), int(rng.integers(1, 4)))
        s = np.round(rng.random(n), int(rng.integers(1, 4)))
        for site in (np.zeros(n, int), np.full(n, "one"), np.ones(n)):
            r = within_site_c(y, s, site)
            assert abs(r["macro_somers_d"] - somers_d(y, s)) < 1e-12
            assert abs(r["macro_c"] - (somers_d(y, s) + 1) / 2) < 1e-12


def test_single_site_multilabel_matches_per_label_somers_d():
    rng = np.random.default_rng(17)
    Y, S = rng.random((150, 4)), np.round(rng.random((150, 4)), 2)
    r = within_site_c(pd.DataFrame(Y, columns=list("abcd")),
                      pd.DataFrame(S, columns=list("abcd")), np.zeros(150, int))
    for j, lb in enumerate("abcd"):
        assert abs(r["per_label"][lb]["somers_d"] - somers_d(Y[:, j], S[:, j])) < 1e-12
    assert abs(r["macro_somers_d"]
               - np.mean([2 * r["per_label"][lb]["c"] - 1 for lb in "abcd"])) < 1e-12


def test_more_sites_never_add_pairs():
    """Splitting a population into sites can only remove pairs."""
    rng = np.random.default_rng(19)
    y, s = rng.random(300), rng.random(300)
    one = within_site_c(y, s, np.zeros(300, int))["total_pairs"]
    many = within_site_c(y, s, rng.integers(0, 9, 300))["total_pairs"]
    assert many < one


# ===========================================================================
# paired_bootstrap
# ===========================================================================
def test_paired_bootstrap_is_seed_deterministic_and_seed_dependent():
    rng = np.random.default_rng(23)
    y = (rng.random((80, 3)) > 0.5).astype(float)
    a = y * 0.4 + 0.6 * rng.random((80, 3))
    b = rng.random((80, 3))
    r1 = paired_bootstrap(y, a, b, n_boot=200, seed=1)
    r2 = paired_bootstrap(y, a, b, n_boot=200, seed=1)
    r3 = paired_bootstrap(y, a, b, n_boot=200, seed=2)
    assert r1 == r2
    assert r1["ci_lo"] != r3["ci_lo"]
    assert r1["delta"] == r3["delta"]        # the point estimate is not random
    assert r1["seed"] == 1 and r1["n_boot"] == 200 and r1["n"] == 80


def test_paired_bootstrap_requires_n_boot_and_seed():
    y = np.array([0., 1., 0., 1.])
    try:
        paired_bootstrap(y, y, y)            # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("n_boot and seed must be required")


def test_paired_bootstrap_delta_is_the_observed_difference():
    rng = np.random.default_rng(29)
    y = (rng.random(120) > 0.5).astype(float)
    a = y * 0.5 + 0.5 * rng.random(120)
    b = rng.random(120)
    r = paired_bootstrap(y, a, b, n_boot=50, seed=4)
    assert abs(r["delta"] - ((somers_d(y, a) - somers_d(y, b)) / 2)) < 1e-12
    assert abs(r["c_a"] - _auc_rank(y, a)) < 1e-12
    assert abs(r["c_b"] - _auc_rank(y, b)) < 1e-12


def test_paired_bootstrap_identical_scorers_give_zero():
    rng = np.random.default_rng(31)
    y = (rng.random(60) > 0.5).astype(float)
    a = rng.random(60)
    r = paired_bootstrap(y, a, a, n_boot=100, seed=5)
    assert r["delta"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0
    assert abs(r["r"] - 1.0) < 1e-9


def test_paired_bootstrap_degenerate_returns_nan():
    y = np.ones(20)                      # one class: every replicate undefined
    a, b = np.arange(20.), np.arange(20.)[::-1].copy()
    r = paired_bootstrap(y, a, b, n_boot=50, seed=6)
    assert np.isnan(r["delta"]) and np.isnan(r["ci_lo"]) and np.isnan(r["ci_hi"])
    assert r["n_boot"] == 0 and r["n_boot_requested"] == 50
    r0 = paired_bootstrap(np.array([0., 1.]), np.array([0., 1.]),
                          np.array([1., 0.]), n_boot=0, seed=7)
    assert r0["n_boot"] == 0 and np.isnan(r0["ci_lo"])
    # `delta` is on the c scale, which is the scale every 58-study number in
    # this project is quoted on: a perfect ranker is 1.0, its inversion is 0.0.
    # Somers' D is twice that, and both are asserted here so the scale cannot be
    # confused again.
    assert r0["delta"] == 1.0                              # c(a) - c(b)
    assert r0["somers_d_a"] - r0["somers_d_b"] == 2.0      # the D scale


# ===========================================================================
# no side effects
# ===========================================================================
def test_module_is_silent_and_writes_nothing():
    before = sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))))
    rng = np.random.default_rng(37)
    y = (rng.random((40, 2)) > 0.5).astype(float)
    a, b = rng.random((40, 2)), rng.random((40, 2))
    buf = io.StringIO()
    with redirect_stdout(buf):
        somers_d(y[:, 0], a[:, 0])
        within_site_c(y, a, rng.integers(0, 3, 40))
        paired_bootstrap(y, a, b, n_boot=20, seed=41)
        somers_d(np.ones(4), np.arange(4.))          # the degenerate paths too
        within_site_c(np.ones((4, 1)), np.zeros((4, 1)), np.zeros(4, int))
    assert buf.getvalue() == ""
    assert sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))) == before


def test_inputs_are_not_mutated():
    rng = np.random.default_rng(43)
    y = rng.random((50, 2))
    a, b = rng.random((50, 2)), rng.random((50, 2))
    site = rng.integers(0, 4, 50)
    y0, a0, b0, s0 = y.copy(), a.copy(), b.copy(), site.copy()
    within_site_c(y, a, site)
    paired_bootstrap(y, a, b, n_boot=20, seed=47)
    somers_d(y[:, 0], a[:, 0])
    assert np.array_equal(y, y0) and np.array_equal(a, a0)
    assert np.array_equal(b, b0) and np.array_equal(site, s0)


# ===========================================================================
# 3. paired_58 / verdict_58 on synthetic results
# ===========================================================================
def _fake_result(**over):
    base = dict(delta=0.01, boot_mean=0.011, ci_lo=-0.01, ci_hi=0.03,
                r_paired=0.75, r_by_label={"ACL": 0.75}, mdd_realized=2.80 * 0.01,
                n_boot=100, seed=1, c_a=0.71, c_b=0.70, se=0.01, r_boot=0.9,
                n=58, n_labels_used=1, labels=("ACL",), name_a="a", name_b="b",
                per_label=())
    base.update(over)
    return Paired58Result(**base)


def test_paired_58_result_is_frozen():
    r = _fake_result()
    for field in ("delta", "r_paired", "mdd_realized"):
        try:
            setattr(r, field, 0.0)
        except Exception as exc:                                # noqa: BLE001
            assert type(exc).__name__ == "FrozenInstanceError", exc
        else:
            raise AssertionError(f"{field} was assignable")


def test_paired_58_result_rejects_an_mdd_that_did_not_come_from_the_se():
    """The field cannot be filled in from a table.  This is the check that
    stops the accounting from being decorative."""
    try:
        _fake_result(mdd_realized=0.05)
    except ValueError as exc:
        assert "mdd_realized" in str(exc)
    else:
        raise AssertionError("a mismatched mdd_realized was accepted")


def test_paired_58_result_rejects_an_inverted_interval_and_a_bad_r():
    for over, token in [({"ci_lo": 0.05, "ci_hi": -0.05}, "ci_lo"),
                        ({"r_paired": 1.7}, "r_paired"),
                        ({"r_boot": -2.0}, "r_boot"),
                        ({"n_boot": -1}, "n_boot")]:
        try:
            _fake_result(**over)
        except ValueError as exc:
            assert token in str(exc), (over, exc)
        else:
            raise AssertionError(f"{over} was accepted")


def test_verdict_58_refuses_anything_but_a_result():
    """A dict with the right keys is exactly the thing that must not work."""
    r = _fake_result()
    ok = verdict_58(r)
    assert isinstance(ok, str) and "measured pair correlation" in ok
    for bad in (r.to_dict(), (0.01, -0.01, 0.03), 0.01, None,
                {"delta": 0.01, "r_paired": 0.75, "mdd_realized": 0.028}):
        try:
            verdict_58(bad)
        except TypeError as exc:
            assert "Paired58Result" in str(exc)
        else:
            raise AssertionError(f"verdict_58 accepted {type(bad).__name__}")


def test_verdict_58_always_prints_r_and_the_mdd():
    """ADR-030: both must appear before anything is said about the 58."""
    for over in ({}, {"delta": 0.20, "ci_lo": 0.15, "ci_hi": 0.25}):
        s = verdict_58(_fake_result(**over))
        assert "measured pair correlation" in s
        assert "minimum detectable difference" in s
        assert "0.7500" in s and "0.0280" in s


def test_paired_58_degenerate_returns_nan_not_an_exception():
    y = np.zeros((8, 2))                    # one class only: c undefined
    a, b = np.random.default_rng(0).normal(size=(8, 2)), np.zeros((8, 2))
    r = paired_58(y, a, b, n_boot=10, seed=0, labels=["x", "y"])
    assert not np.isfinite(r.delta) and not np.isfinite(r.mdd_realized)
    assert not r.resolved and not r.excludes_zero
    assert isinstance(verdict_58(r), str)


def test_paired_58_labels_travel_into_r_by_label():
    G = np.array([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    A = np.array([[.1, .9], [.8, .2], [.7, .6], [.2, .3]])
    B = A[:, ::-1].copy()
    r = paired_58(G, A, B, n_boot=20, seed=3, labels=["left", "right"])
    assert set(r.r_by_label) == {"left", "right"}
    assert r.labels == ("left", "right")
    r2 = paired_58(pd.DataFrame(G, columns=["p", "q"]), A, B, n_boot=20, seed=3)
    assert set(r2.r_by_label) == {"p", "q"}


# ===========================================================================
if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
