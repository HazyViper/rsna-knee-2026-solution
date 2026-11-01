"""Evaluation metrics for the RSNA knee project.  Single source of truth.

Every selection decision in this project rides on three numbers:

    within-stratum concordance c   the cheap, high-resolution sieve
    Somers' D = 2c - 1             the same quantity on a signed scale
    a paired bootstrap on the 58   the expensive confirmation against truth

They used to be spelled out in eleven places.  They are spelled out here now,
once, so that the arithmetic can be read and checked without reading the
analysis scripts that consume it.

PUBLIC API
----------
    somers_d(y_true, y_score)                                 -> float
    within_site_c(y_true, y_score, site, folds, fold_mask)    -> dict
    paired_bootstrap(y_true, score_a, score_b, n_boot, seed)  -> dict
    paired_58(y_true, score_a, score_b, n_boot, seed, ...)    -> Paired58Result
    verdict_58(result)                                        -> str

THE 58-STUDY ACCOUNTING IS NOT OPTIONAL
---------------------------------------
`paired_bootstrap` returns a dict, and a caller is free to quote `delta` out of
it and stop.  Twice now -- phase 9 step 5 and phase 10-A -- the sieve (c) and
the confirmation (the 58) pointed opposite ways and the session stopped at
"not refuted, not resolved", because ADR-030's accounting (the MEASURED pair
correlation and the REALISED minimum detectable difference) was a separate
computation somebody had to remember to start.  It was never the judgement
that was loose; it was that the arithmetic had to be re-launched by hand.

So `paired_58` returns a frozen `Paired58Result` whose TYPE carries `r_paired`
and `mdd_realized`.  They cannot be dropped on the way to a sentence, because
`verdict_58` renders the sentence and accepts nothing else -- not a dict, not a
tuple of numbers assembled by the caller.  What that buys is narrow and worth
stating plainly: it cannot stop anyone from typing a claim about the 58 into a
report by hand.  It stops a report GENERATOR from formatting one out of numbers
that never went through the accounting.

Everything else is private (leading underscore).  The project's three legacy
hubs -- `scripts_p6/p6_common.py`, `scripts_p8/p8c_metric.py` and
`scripts_p8/p8b_metric.py` -- import the private helpers deliberately so that
the arithmetic exists exactly once; they add naming, warnings and I/O, never
a formula.  Every phase 6, 7 and 8 script reaches this file through one of
those three, so no analysis script had to be edited to be brought onto it.

TWO IMPLEMENTATIONS THAT ARE NOT FOLDED IN, AND WHY
---------------------------------------------------
  * `p8b_metric.pooled_concordance` keeps the 8-B denominator, C / (C + D).
    That is Goodman-Kruskal's gamma, not Somers' D, and it is WRONG -- but
    `phase8b_report.md` was published from it, so it stays reproducible.  Only
    its pair COUNTING is delegated here; the denominator stays spelled out in
    that file, next to the note saying so.
  * `scripts_p2/*.py` each carry a private `def auc(y, s)` that casts the target
    with `np.asarray(y, bool)` and applies no finite filter.  On finite binary
    input it agrees with `_auc_binary` exactly (verified on three fixtures over
    all 18 copies in the tree); on input containing NaN it does not, because
    `np.asarray(np.nan, bool)` is True and a missing label becomes a positive.
    Those scripts ran once in 2026-08 and their mtimes are this repository's
    only provenance (there is no git), so they are left untouched and recorded
    here instead.

PROPERTIES OF THIS MODULE
-------------------------
  * No side effects.  Nothing is read, written, printed, cached or mutated at
    module scope.  Warnings are returned as counts, not emitted.
  * Randomness enters only through the `seed` argument, which has no default.
    Two calls with the same seed and the same data give the same answer.
  * Degenerate input returns nan, never an exception.  See each docstring.
  * Dependencies: numpy.  pandas objects are accepted (a DataFrame supplies
    its column names as label names) but pandas is never imported.

COST
----
The pair kernel is exact and O(n^2) per label; there is no rank-based fast
path, because a second code path is a second thing to verify.  On the two
populations this project actually uses -- 4,402 OOF studies split into 51
strata, and the 58 gold studies -- that is a fraction of a second.
`paired_bootstrap` costs O(n_boot * n_labels * n^2) and is meant for the 58.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["somers_d", "within_site_c", "paired_bootstrap",
           "Paired58Result", "paired_58", "verdict_58"]

_BLOCK = 512          # row-blocking width of the pair kernel; affects speed only

# Minimum detectable difference = _MDD_Z * SE(difference): two-sided 95% with
# 80% power, z_{0.975} + z_{0.80} = 1.959964 + 0.841621 = 2.801585, quoted as
# 2.80 throughout this project.  It is the k = 1 case of ADR-026's threshold
# (z_{1-alpha/(2k)} + 0.84) * SE and the constant used in p9_4_conflict.py:112,
# whose published 0.0657 this module must reproduce.  A family correction
# raises the multiplier; it never lowers it.
_MDD_Z = 2.80


# ===========================================================================
# input coercion
# ===========================================================================
def _as_2d(x: Any, name: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """(n,) or (n, L) -> float (n, L), plus label names if the object carries
    them (a pandas DataFrame's columns, a Series' name).  Never copies names
    out of thin air; returns None when the object has none."""
    labels: Optional[List[str]] = None
    cols = getattr(x, "columns", None)
    if cols is not None:
        labels = [str(c) for c in cols]
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        nm = getattr(x, "name", None)
        if labels is None and nm is not None:
            labels = [str(nm)]
        a = a[:, None]
    elif a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {a.shape}")
    return a, labels


def _as_1d(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def _resolve_labels(labels: Optional[List[str]], n_cols: int) -> List[str]:
    if labels is not None and len(labels) == n_cols:
        return labels
    return [str(j) for j in range(n_cols)]


# ===========================================================================
# the pair kernel -- the only place pairs are counted
# ===========================================================================
def _pair_counts(y: np.ndarray, s: np.ndarray) -> Tuple[int, int, int]:
    """Concordant, discordant and score-tied pair counts over the pairs whose
    TARGET differs.  Both inputs must already be finite.

    sign(y_i - y_j) * sign(s_i - s_j) is symmetric with a zero diagonal, so the
    full matrix counts every unordered pair twice and the diagonal contributes
    nothing.  Row blocking keeps peak memory at _BLOCK * n rather than n^2.

    Returns (C, D, T_s) where T_s is the pairs with an untied target and a tied
    score.  C + D + T_s is the count of untied-target pairs, which depends on
    the target alone -- never on the model."""
    n = len(y)
    if n < 2:
        return 0, 0, 0
    C = D = NY = 0
    for i in range(0, n, _BLOCK):
        dy = np.sign(y[i:i + _BLOCK, None] - y[None, :])
        ds = np.sign(s[i:i + _BLOCK, None] - s[None, :])
        p = dy * ds
        C += int((p > 0).sum())
        D += int((p < 0).sum())
        NY += int((dy != 0).sum())
    C, D, NY = C // 2, D // 2, NY // 2
    return C, D, NY - C - D


def _c_from_counts(c: float, d: float, t: float) -> float:
    """c = (C + 0.5 * T_s) / (C + D + T_s).

    THE DENOMINATOR INCLUDES TIED-PREDICTION PAIRS.  Goodman-Kruskal's gamma,
    C / (C + D), drops them, and this project used gamma by mistake through
    phase 8-B.  The difference is not cosmetic: gamma's denominator depends on
    the model, so a model that emits few distinct values is graded only on the
    pairs it happens to have an opinion about, and is flattered for it.  Here
    the denominator is a property of the target and the strata alone, identical
    for every model scored on the same rows.

    Consequences: Somers' D = 2c - 1 holds exactly; a constant model scores
    exactly 0.5 (gamma leaves it undefined); with one site and a binary target
    this is exactly ROC-AUC, ties and all."""
    tot = c + d + t
    if tot <= 0:
        return float("nan")
    return float((c + 0.5 * t) / tot)


def _finite_pair(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(s)


def _c_one(y: np.ndarray, s: np.ndarray) -> float:
    """Single-stratum concordance on already-finite input."""
    return _c_from_counts(*_pair_counts(y, s))


def _macro_c(Y: np.ndarray, S: np.ndarray) -> float:
    """Mean of the per-column concordance over the columns where it is defined.
    Columns with no untied-target pair are excluded, not folded in as 0.5.
    Returns nan when no column is defined."""
    vals = []
    for j in range(Y.shape[1]):
        m = _finite_pair(Y[:, j], S[:, j])
        v = _c_one(Y[m, j], S[m, j])
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ===========================================================================
# the legacy binary-target AUC convention
# ===========================================================================
def _auc_binary(y: Any, s: Any) -> float:
    """ROC-AUC under the convention this project has used since phase 6:
    positives are the rows with y == 1, negatives the rows with y == 0, ties
    take half credit, and any OTHER target value still takes part in the
    ranking while belonging to neither class.

    For a strictly binary target this equals `(somers_d(y, s) + 1) / 2` exactly
    (asserted in tests/test_metrics.py).  For a continuous target it does NOT,
    and callers that pass pseudo-label scores as the target -- a0b_bottleneck,
    a1_eval, a3c_metadata_corpus, p8_site -- are relying on this convention,
    which is why it survives as its own function instead of being folded into
    `somers_d`.

    Returns nan if either class is empty after non-finite rows are dropped."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    m = _finite_pair(y, s)
    y, s = y[m], s[m]
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _boot_auc_ci(y: Any, s: Any, n: int, seed: int) -> Tuple[float, float]:
    """Percentile bootstrap CI for one scorer's `_auc_binary`, resampling rows.

    Backs `scripts_p6/p6_common.auc_boot`, whose published numbers this must
    reproduce bit for bit: one `rng.integers(0, n_rows, n_rows)` draw per
    replicate, non-finite replicates dropped, 2.5 / 97.5 percentiles.
    Returns (nan, nan) when no replicate is defined."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    out: List[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = _auc_binary(y[idx], s[idx])
        if np.isfinite(a):
            out.append(a)
    if not out:
        return (float("nan"), float("nan"))
    arr = np.sort(np.asarray(out))
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def _boot_auc_diff(y: Any, s_a: Any, s_b: Any, n: int,
                   seed: int) -> Tuple[float, float, float]:
    """Paired bootstrap of `_auc_binary(a) - _auc_binary(b)` on ONE label,
    common random numbers.  Backs `scripts_p6/p6_common.auc_diff_boot`.

    Returns (mean of the replicate differences, 2.5 pct, 97.5 pct).  Note the
    first element is the BOOTSTRAP MEAN, not the difference observed on the full
    sample; `paired_bootstrap` returns the observed difference as `delta` and
    the bootstrap mean separately as `boot_mean`.  The two differ by the
    bootstrap bias, which is small but not zero.  Kept as-is because phase 6-8
    numbers were reported from it."""
    y = np.asarray(y, dtype=float)
    s_a = np.asarray(s_a, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    rng = np.random.default_rng(seed)
    out: List[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a, b = _auc_binary(y[idx], s_a[idx]), _auc_binary(y[idx], s_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            out.append(a - b)
    if not out:
        return (float("nan"),) * 3
    arr = np.sort(np.asarray(out))
    return (float(np.mean(arr)), float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)))


# ===========================================================================
# row selection
# ===========================================================================
def _row_selection(n: int, folds: Any, fold_mask: Any) -> np.ndarray:
    """Boolean row mask from the (folds, fold_mask) pair.

    folds=None, fold_mask=None      every row
    folds=None, fold_mask=bool[n]   that row mask, used verbatim
    folds=f,    fold_mask=None      every row (every fold)
    folds=f,    fold_mask=bool[n]   that row mask
    folds=f,    fold_mask=ids       rows whose fold id is in `ids`

    There is deliberately no default fold set.  A caller that wants fold 0 has
    to say fold 0."""
    if folds is None:
        if fold_mask is None:
            return np.ones(n, dtype=bool)
        fm = np.asarray(fold_mask)
        if fm.dtype != np.bool_ or fm.shape != (n,):
            raise ValueError(
                "fold_mask without folds must be a boolean row mask of length "
                f"{n}; got dtype {fm.dtype} shape {fm.shape}.  Pass `folds` to "
                "select by fold id.")
        return fm
    f = np.asarray(folds)
    if f.shape != (n,):
        raise ValueError(f"folds must have length {n}, got shape {f.shape}")
    if fold_mask is None:
        return np.ones(n, dtype=bool)
    fm = np.asarray(fold_mask)
    if fm.dtype == np.bool_:
        if fm.shape != (n,):
            raise ValueError(
                f"a boolean fold_mask must have length {n}, got {fm.shape}")
        return fm
    wanted = set(np.atleast_1d(fm).ravel().tolist())
    return np.isin(f, list(wanted))


# ===========================================================================
# public API
# ===========================================================================
def somers_d(y_true: Any, y_score: Any) -> float:
    """Somers' D of `y_score` with respect to `y_true`, in [-1, +1].

    D = (C - D) / (C + D + T_s) = 2c - 1, where the pairs counted are those
    whose TARGET differs, and T_s is the subset of those whose PREDICTION is
    tied.

    THE DENOMINATOR INCLUDES TIED-PREDICTION PAIRS.  This is what separates
    Somers' D from Goodman-Kruskal's gamma, which is (C - D) / (C + D) and
    excludes them.  Gamma's denominator is a function of the model, so it
    rewards a model for emitting coarse scores; D's is a function of the target
    alone and is identical for every model scored on the same rows.  This
    project computed gamma by mistake through phase 8-B; see
    reports/phase8c_report.md section C-1.

    The target may be continuous; it does not have to be binary.  With a binary
    target, (D + 1) / 2 is exactly ROC-AUC including the half credit for ties.

    Returns nan, never raises, when:
      * fewer than two rows survive the non-finite filter;
      * every target value is equal, or only one class is present -- in both
        cases there is no untied-target pair and the statistic is undefined.
    Returns 0.0, not nan, when every prediction is equal: that is a genuine
    measurement of no discrimination, and it is the case gamma gets wrong.
    Rows where either input is NaN or infinite are dropped pairwise.

    Args:
        y_true:  (n,) target.  Continuous or binary.
        y_score: (n,) prediction, higher = more positive.

    Returns:
        float in [-1, +1], or nan.
    """
    y = _as_1d(y_true, "y_true")
    s = _as_1d(y_score, "y_score")
    if y.shape != s.shape:
        raise ValueError(f"y_true {y.shape} and y_score {s.shape} differ")
    m = _finite_pair(y, s)
    c = _c_one(y[m], s[m])
    return float("nan") if not np.isfinite(c) else float(2.0 * c - 1.0)


def within_site_c(y_true: Any, y_score: Any, site: Any,
                  folds: Any = None, fold_mask: Any = None) -> Dict[str, Any]:
    """Within-site concordance c: pool the within-site pairs, then divide once.

    Only pairs of studies from the SAME site are counted, which removes every
    pair a model could win by telling the sites apart rather than by reading
    the knee.  The pooling order matters: all sites contribute to one numerator
    and one denominator, and the division happens once at the end.  Averaging
    the per-site c values instead would give a ten-study site the same weight
    as a five-hundred-study one.

    The scale is c (the AUC scale, 0.5 = chance).  `somers_d` = 2c - 1 is
    returned alongside at both the macro and the per-label level, and uses the
    same tied-prediction-inclusive denominator as `somers_d` above.

    Fold selection is an argument and has no default: pass `folds` (a per-row
    fold id) together with `fold_mask` (the fold ids to keep), or pass
    `fold_mask` alone as a boolean row mask.  Passing neither uses every row.

    Degenerate input returns nan rather than raising:
      * a label with no untied-target pair inside any site has c = nan and is
        EXCLUDED from the macro (it is named in `undefined_labels`, not folded
        in as 0.5);
      * if no label is defined, `macro_c` is nan;
      * a site with fewer than two surviving rows contributes no pairs;
      * rows where the target or the score is NaN or infinite are dropped, per
        label, and the surviving count is reported as `per_label[...]["n"]`.

    Args:
        y_true:   (n,) or (n, L) target.  A DataFrame supplies label names.
        y_score:  (n,) or (n, L) prediction, same shape as y_true.
        site:     (n,) site key.  Any hashable dtype; NaN is a key like any
                  other, so map missing sites to a sentinel before calling if
                  you do not want them pooled together.
        folds:    (n,) fold id per row, or None.
        fold_mask: fold ids to keep, or a boolean row mask, or None.

    Returns:
        dict with
          macro_c, macro_somers_d : float (nan if no label is defined)
          n                       : int, rows selected by (folds, fold_mask)
          n_sites                 : int, distinct site keys among those rows
          labels                  : list[str]
          n_labels_used           : int
          undefined_labels        : list[str]
          total_pairs             : int, summed over labels
          per_label               : {label: {c, somers_d, n, n_sites,
                                             concordant, discordant,
                                             tied_score, n_pairs}}
    """
    Y, lab_y = _as_2d(y_true, "y_true")
    S, lab_s = _as_2d(y_score, "y_score")
    if Y.shape != S.shape:
        raise ValueError(f"y_true {Y.shape} and y_score {S.shape} differ")
    labels = _resolve_labels(lab_y if lab_y is not None else lab_s, Y.shape[1])

    site_arr = np.asarray(site)
    if site_arr.ndim != 1 or site_arr.shape[0] != Y.shape[0]:
        raise ValueError(
            f"site must be 1-D of length {Y.shape[0]}, got {site_arr.shape}")

    keep = _row_selection(Y.shape[0], folds, fold_mask)
    Y, S, site_arr = Y[keep], S[keep], site_arr[keep]

    per: Dict[str, Dict[str, Any]] = {}
    for j, lb in enumerate(labels):
        y, s = Y[:, j], S[:, j]
        m = _finite_pair(y, s)
        yk, sk, kk = y[m], s[m], site_arr[m]
        C = D = T = 0
        keys = np.unique(kk) if len(kk) else np.empty(0)
        for k in keys:
            sub = kk == k
            c_k, d_k, t_k = _pair_counts(yk[sub], sk[sub])
            C += c_k
            D += d_k
            T += t_k
        c = _c_from_counts(C, D, T)
        per[lb] = {"c": c,
                   "somers_d": (2.0 * c - 1.0) if np.isfinite(c) else float("nan"),
                   "n": int(m.sum()), "n_sites": int(len(keys)),
                   "concordant": int(C), "discordant": int(D),
                   "tied_score": int(T), "n_pairs": int(C + D + T)}

    vals = [per[lb]["c"] for lb in labels if np.isfinite(per[lb]["c"])]
    macro = float(np.mean(vals)) if vals else float("nan")
    return {"macro_c": macro,
            "macro_somers_d": (2.0 * macro - 1.0) if np.isfinite(macro)
                              else float("nan"),
            "n": int(keep.sum()),
            "n_sites": int(len(np.unique(site_arr))) if len(site_arr) else 0,
            "labels": list(labels),
            "n_labels_used": len(vals),
            "undefined_labels": [lb for lb in labels
                                 if not np.isfinite(per[lb]["c"])],
            "total_pairs": int(sum(per[lb]["n_pairs"] for lb in labels)),
            "per_label": per}


def paired_bootstrap(y_true: Any, score_a: Any, score_b: Any,
                     n_boot: int, seed: int) -> Dict[str, Any]:
    """Paired bootstrap over STUDIES of c(a) - c(b), common random numbers.

    One multinomial draw per replicate is applied to both scorers -- same rows,
    same order -- so the two are never compared across different resamples.
    That is what makes the difference CI narrower than either absolute CI, and
    the narrowing is real only to the extent the two scorers correlate; the
    measured correlation is returned as `r` so the caller can say so rather
    than assume it.

    The statistic is the concordance c of `somers_d` above, macro-averaged over
    the columns where it is defined.  With a binary target that is macro
    ROC-AUC, which is the scale every 58-study number in this project is
    quoted on.

    `delta` is the difference OBSERVED on the full sample, not the bootstrap
    mean; the bootstrap supplies the interval, not the point estimate.

    `n_boot` and `seed` are required and have no defaults.  Randomness enters
    only through `seed`.

    Degenerate input returns nan rather than raising:
      * a replicate in which either scorer has no defined column is dropped;
        `n_boot` in the return value is the number of replicates actually
        used, which can be smaller than the number requested;
      * if no replicate survives, ci_lo / ci_hi / se / r are nan;
      * a column with no untied-target pair is excluded from that replicate's
        macro rather than folded in as 0.5;
      * rows where the target or a score is NaN or infinite are dropped per
        column.
    Resampling a study twice is harmless: the duplicate pairs with itself on a
    tied target, and tied-target pairs are outside the denominator.

    Cost is O(n_boot * n_labels * n^2).  Sized for the 58 gold studies.

    Args:
        y_true:  (n,) or (n, L) target.  A DataFrame supplies label names.
        score_a: (n,) or (n, L) prediction, same shape as y_true.
        score_b: (n,) or (n, L) prediction, same shape as y_true.
        n_boot:  number of bootstrap replicates.  Required.
        seed:    seed for numpy.random.default_rng.  Required.

    Returns:
        dict with
          delta          : float, c(a) - c(b) on the full sample
          ci_lo, ci_hi   : float, 2.5 / 97.5 percentiles of the paired
                           difference across replicates
          n_boot         : int, replicates actually used
          seed           : int, echoed back
          c_a, c_b       : float, the two full-sample macro c values
          somers_d_a, somers_d_b : float, the same on the 2c - 1 scale
          se             : float, sd of the paired difference
          r              : float, correlation of the two bootstrap
                           distributions -- the paired narrowing is
                           1 - sqrt(2 * (1 - r)) when it is positive
          n              : int, rows
          n_boot_requested : int
          boot_mean      : float, mean of the paired difference (a bias check
                           against `delta`, not a replacement for it)
    """
    if int(n_boot) != n_boot or n_boot < 0:
        raise ValueError(f"n_boot must be a non-negative integer, got {n_boot!r}")
    n_boot = int(n_boot)

    Y, _ = _as_2d(y_true, "y_true")
    A, _ = _as_2d(score_a, "score_a")
    B, _ = _as_2d(score_b, "score_b")
    if not (Y.shape == A.shape == B.shape):
        raise ValueError(f"shapes differ: y_true {Y.shape}, score_a {A.shape}, "
                         f"score_b {B.shape}")
    n = Y.shape[0]

    c_a, c_b = _macro_c(Y, A), _macro_c(Y, B)
    delta = float(c_a - c_b) if np.isfinite(c_a) and np.isfinite(c_b) \
        else float("nan")

    rng = np.random.default_rng(seed)
    da: List[float] = []
    db: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n) if n else np.empty(0, dtype=int)
        va, vb = _macro_c(Y[idx], A[idx]), _macro_c(Y[idx], B[idx])
        if np.isfinite(va) and np.isfinite(vb):
            da.append(va)
            db.append(vb)

    if da:
        av, bv = np.asarray(da), np.asarray(db)
        d = av - bv
        ci_lo = float(np.percentile(d, 2.5))
        ci_hi = float(np.percentile(d, 97.5))
        se = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
        r = float(np.corrcoef(av, bv)[0, 1]) if len(d) > 2 else float("nan")
        boot_mean = float(np.mean(d))
    else:
        ci_lo = ci_hi = se = r = boot_mean = float("nan")

    return {"delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_boot": len(da), "seed": seed,
            "c_a": c_a, "c_b": c_b,
            "somers_d_a": (2.0 * c_a - 1.0) if np.isfinite(c_a) else float("nan"),
            "somers_d_b": (2.0 * c_b - 1.0) if np.isfinite(c_b) else float("nan"),
            "se": se, "r": r, "n": n,
            "n_boot_requested": n_boot, "boot_mean": boot_mean}


# ===========================================================================
# the 58-study accounting
# ===========================================================================
def _corr(a: Any, b: Any) -> float:
    """Pearson correlation of two arrays, flattened, non-finite cells dropped
    pairwise.  nan when fewer than three cells survive or either side is
    constant.  On all-finite input this is `np.corrcoef(a.ravel(), b.ravel())`,
    which is what p9_4_conflict.py:96-97 published."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    x, y = x[m], y[m]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@dataclass(frozen=True)
class Paired58Result:
    """The result of a paired comparison on the 58 gold studies, WITH the
    accounting ADR-030 requires before the comparison may be written down.

    The nine fields the accounting is made of come first.  Three of them are
    the ones that keep going missing when this is done by hand:

      delta        the difference OBSERVED on the full sample.
      boot_mean    the mean of the bootstrap replicates.  A DIFFERENT NUMBER.
                   `p8c_metric.boot_diff` returns only the second and calls it
                   `diff_mean`; phases 8-C..8-H quoted the observed difference
                   as their point estimate, so no past verdict moves, but the
                   two are measured to differ by +0.000147 on the sieve
                   (docs/STATE.md open item 1) and this type refuses to let a
                   caller confuse them.
      r_paired     the MEASURED correlation between the two arms' predictions,
                   pooled over labels.  Pitfall 7: r is not knowable in advance
                   -- 8-E saw 0.63 between models whose inputs were identical
                   to the pixel -- so it is measured and reported, never
                   assumed.  Compare against the project's yardsticks: fold
                   siblings 0.944-0.960, M1-M2 0.748, M2-M3 0.894.
      mdd_realized the smallest difference this comparison could have resolved,
                   2.80 * SE(difference).  A |delta| inside it means the 58
                   said nothing, in either direction.

    NOTE THE TWO CORRELATIONS.  `r_paired` is between the two PREDICTION
    vectors.  `r_boot` is between the two arms' bootstrap STATISTIC
    distributions, and it is that one that drives the paired narrowing
    1 - sqrt(2 * (1 - r)).  They answer different questions and they are not
    equal; the project's published "measured pair correlation" is `r_paired`.

    Frozen, and validated on construction: `mdd_realized` must be 2.80 * `se`,
    the interval must be ordered, the correlations must be correlations.  That
    is not ceremony -- it is what stops a hand-assembled instance from being
    passed to `verdict_58` with numbers that were never measured together.
    """
    # --- the accounting, in the order ADR-030 asks for it ------------------
    delta: float
    boot_mean: float
    ci_lo: float
    ci_hi: float
    r_paired: float
    r_by_label: Dict[str, float]
    mdd_realized: float
    n_boot: int
    seed: int
    # --- the context needed to read the accounting -------------------------
    c_a: float
    c_b: float
    se: float
    r_boot: float
    n: int
    n_labels_used: int
    labels: Tuple[str, ...]
    name_a: str
    name_b: str
    per_label: Tuple[Dict[str, Any], ...]

    def __post_init__(self) -> None:
        if int(self.n_boot) != self.n_boot or self.n_boot < 0:
            raise ValueError(f"n_boot must be a non-negative integer, "
                             f"got {self.n_boot!r}")
        if np.isfinite(self.ci_lo) and np.isfinite(self.ci_hi) \
                and self.ci_lo > self.ci_hi:
            raise ValueError(f"ci_lo {self.ci_lo} > ci_hi {self.ci_hi}")
        if np.isfinite(self.se) and np.isfinite(self.mdd_realized):
            want = _MDD_Z * self.se
            if abs(self.mdd_realized - want) > 1e-12 * max(1.0, abs(want)):
                raise ValueError(
                    f"mdd_realized {self.mdd_realized} is not {_MDD_Z} * se "
                    f"({want}); the MDD must come from the same bootstrap as "
                    f"the interval, not from a table")
        for nm, r in [("r_paired", self.r_paired), ("r_boot", self.r_boot)]:
            if np.isfinite(r) and not (-1.0000001 <= r <= 1.0000001):
                raise ValueError(f"{nm} = {r} is not a correlation")

    # ----------------------------------------------------------- readings
    @property
    def excludes_zero(self) -> bool:
        """Whether the percentile interval excludes zero.  In the sandbox this
        is a description of the interval, not a verdict (ADR-045); a verdict
        needs a pre-registration and a family size."""
        return bool(np.isfinite(self.ci_lo) and np.isfinite(self.ci_hi)
                    and not (self.ci_lo <= 0.0 <= self.ci_hi))

    @property
    def resolved(self) -> bool:
        """Whether |delta| reaches the difference this comparison could detect.
        False means the 58 could not have separated the two arms either way --
        which is NOT the same as their having found no difference."""
        return bool(np.isfinite(self.delta) and np.isfinite(self.mdd_realized)
                    and abs(self.delta) >= self.mdd_realized)

    def r_range(self) -> Tuple[float, float]:
        """(min, max) of the per-label correlations, ignoring undefined ones."""
        vals = [v for v in self.r_by_label.values() if np.isfinite(v)]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    def sentence(self) -> str:
        """The one sentence a report may say about this comparison.

        It carries the observed difference, the interval, the measured pair
        correlation and the realised MDD, in that order, because ADR-030 says
        the last two must be printed BEFORE anyone writes that the 58 refused
        something.  When |delta| is inside the MDD the sentence says so instead
        of reporting a direction."""
        lo, hi = self.r_range()
        head = (f"{self.name_a} - {self.name_b} on the {self.n} true-label "
                f"studies: macro c {self.c_a:.4f} vs {self.c_b:.4f}, "
                f"delta {self.delta:+.4f} (bootstrap mean {self.boot_mean:+.4f}), "
                f"95% CI [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}] from "
                f"{self.n_boot} paired replicates at seed {self.seed}; "
                f"measured pair correlation {self.r_paired:.4f} pooled, "
                f"{lo:.3f}-{hi:.3f} by label; "
                f"minimum detectable difference here {self.mdd_realized:.4f}.")
        if not self.resolved:
            return (head + " |delta| is inside that MDD, so the 58 could not "
                    "have resolved a difference of this size in either "
                    "direction and this comparison neither supports nor "
                    "refutes the arm.")
        tail = ("The interval excludes zero" if self.excludes_zero
                else "The interval still includes zero")
        return (head + f" |delta| reaches the MDD. {tail}, and the difference "
                f"runs {'in favour of ' + self.name_a if self.delta > 0 else 'against ' + self.name_a}.")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready.  `sentence` is included so that an artifact carries the
        reading beside the numbers it was made from."""
        lo, hi = self.r_range()
        return {"name_a": self.name_a, "name_b": self.name_b,
                "n": self.n, "labels": list(self.labels),
                "n_labels_used": self.n_labels_used,
                "c_a": self.c_a, "c_b": self.c_b,
                "delta": self.delta, "boot_mean": self.boot_mean,
                "se": self.se, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
                "r_paired": self.r_paired,
                "r_by_label": dict(self.r_by_label),
                "r_by_label_min": lo, "r_by_label_max": hi,
                "r_boot": self.r_boot,
                "mdd_realized": self.mdd_realized,
                "excludes_zero": self.excludes_zero,
                "resolved": self.resolved,
                "n_boot": self.n_boot, "seed": self.seed,
                "per_label": [dict(d) for d in self.per_label],
                "sentence": self.sentence()}


def paired_58(y_true: Any, score_a: Any, score_b: Any, n_boot: int, seed: int,
              labels: Optional[Sequence[str]] = None,
              name_a: str = "a", name_b: str = "b") -> Paired58Result:
    """`paired_bootstrap` plus the accounting, returned as one frozen object.

    The bootstrap is `paired_bootstrap`'s, unchanged and not re-implemented:
    one multinomial draw per replicate applied to both arms, macro concordance
    over the labels where it is defined, percentile interval.  On the 58 gold
    studies the target is binary, so that macro is exactly the macro ROC-AUC
    every 58-study number in this project is quoted on, and the numbers this
    returns are the numbers `p6_common.auc` would give.

    What is added on top is arithmetic that has to travel WITH the difference:
    the measured pair correlations (pooled and per label) and the realised
    minimum detectable difference 2.80 * SE.

    This is for the 58 TRUE labels.  Nothing stops it running against a
    pseudo-label target, but a pseudo-label comparison is a sieve reading, not
    a confirmation, and the sieve is `within_site_c` (ADR-022/ADR-029).

    Args:
        y_true:  (n,) or (n, L) true labels.  A DataFrame supplies label names.
        score_a: (n,) or (n, L) prediction, same shape.
        score_b: (n,) or (n, L) prediction, same shape.
        n_boot:  bootstrap replicates.  Required.
        seed:    seed for numpy.random.default_rng.  Required.
        labels:  label names, overriding any the frames carry.
        name_a, name_b: arm names, used only in `sentence()`.

    Returns:
        Paired58Result.  Degenerate input propagates as nan through the fields
        rather than raising, exactly as `paired_bootstrap` does.
    """
    Y, lab_y = _as_2d(y_true, "y_true")
    A, lab_a = _as_2d(score_a, "score_a")
    B, _ = _as_2d(score_b, "score_b")
    if not (Y.shape == A.shape == B.shape):
        raise ValueError(f"shapes differ: y_true {Y.shape}, score_a {A.shape}, "
                         f"score_b {B.shape}")
    if labels is not None:
        names = [str(x) for x in labels]
        if len(names) != Y.shape[1]:
            raise ValueError(f"labels has {len(names)} entries for "
                             f"{Y.shape[1]} columns")
    else:
        names = _resolve_labels(lab_y if lab_y is not None else lab_a,
                                Y.shape[1])

    bs = paired_bootstrap(Y, A, B, n_boot=n_boot, seed=seed)

    per: List[Dict[str, Any]] = []
    r_lab: Dict[str, float] = {}
    for j, lb in enumerate(names):
        m = _finite_pair(Y[:, j], A[:, j]) & np.isfinite(B[:, j])
        ca = _c_one(Y[m, j], A[m, j])
        cb = _c_one(Y[m, j], B[m, j])
        r_lab[lb] = _corr(A[:, j], B[:, j])
        per.append({"label": lb, "c_a": ca, "c_b": cb,
                    "delta": float(ca - cb) if np.isfinite(ca)
                             and np.isfinite(cb) else float("nan"),
                    "r": r_lab[lb], "n": int(m.sum()),
                    "n_pos": int(np.nansum(Y[:, j] == 1))})

    se = bs["se"]
    return Paired58Result(
        delta=bs["delta"], boot_mean=bs["boot_mean"],
        ci_lo=bs["ci_lo"], ci_hi=bs["ci_hi"],
        r_paired=_corr(A, B), r_by_label=r_lab,
        mdd_realized=float(_MDD_Z * se) if np.isfinite(se) else float("nan"),
        n_boot=int(bs["n_boot"]), seed=int(seed),
        c_a=bs["c_a"], c_b=bs["c_b"], se=se, r_boot=bs["r"],
        n=int(Y.shape[0]),
        n_labels_used=int(sum(1 for d in per if np.isfinite(d["delta"]))),
        labels=tuple(names), name_a=str(name_a), name_b=str(name_b),
        per_label=tuple(per))


def verdict_58(result: Any) -> str:
    """Render the sentence a report is allowed to say about the 58.

    Accepts a `Paired58Result` and nothing else.  A dict of numbers the caller
    assembled -- even one holding the right keys -- is refused, because the
    point of the type is that the correlation and the MDD were computed from
    the same bootstrap as the interval rather than looked up afterwards.
    """
    if not isinstance(result, Paired58Result):
        raise TypeError(
            f"verdict_58 needs a Paired58Result, got {type(result).__name__}. "
            f"Build it with paired_58(...): a 58-study reading may not be "
            f"written from numbers that did not travel with the measured pair "
            f"correlation and the realised MDD (ADR-030).")
    return result.sentence()


# ===========================================================================
# block-diagonal machinery for bootstrapping `within_site_c` itself
# ===========================================================================
# `paired_bootstrap` re-counts pairs from scratch on every replicate, which is
# the right trade at n = 58 and hopeless at n = 4,402.  The phase-8 judge
# scripts instead pre-compute the pair indicator matrices once, block-diagonal
# by site, and evaluate a replicate as a quadratic form.  That is an
# optimisation of the SAME arithmetic, so it lives beside it here rather than
# in the scripts.  It stays private: the public surface is the three functions
# above.

def _site_pair_blocks(Y: np.ndarray, S: np.ndarray, site: Any,
                      n_labels: int, max_site: int = 1200
                      ) -> List[List[Tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]]]:
    """Per label, the concordant / discordant / score-tied indicator matrices,
    one small matrix per site instead of one (n, n).

    Only within-site pairs can contribute, so a dense (n, n) form would be
    mostly zeros: sum_k n_k^2 is ~1M cells at the phase-8 rung against ~19M for
    n = 4,407, which is the difference between 40 MB and 700 MB across 12
    labels, and the reason a bootstrap on the full OOF is possible at all.

    `max_site` refuses to build a block bigger than that; a single global
    stratum at n = 4,407 must use a different route."""
    site_arr = np.asarray(site)
    blocks: List[np.ndarray] = []
    for k in np.unique(site_arr):
        idx = np.flatnonzero(site_arr == k)
        if len(idx) < 2:
            continue
        if len(idx) > max_site:
            raise MemoryError(
                f"site {k!r} has {len(idx)} rows > max_site={max_site}; the "
                f"block-diagonal bootstrap would need {len(idx) ** 2} cells "
                f"per label.  Raise max_site deliberately or use a coarser "
                f"population.")
        blocks.append(idx)
    out = []
    for j in range(n_labels):
        per = []
        for idx in blocks:
            yy, ss = Y[idx, j], S[idx, j]
            dy = np.sign(yy[:, None] - yy[None, :])
            ds = np.sign(ss[:, None] - ss[None, :])
            p = dy * ds
            per.append((idx, (p > 0).astype(np.float32),
                        (p < 0).astype(np.float32),
                        ((dy != 0) & (ds == 0)).astype(np.float32)))
        out.append(per)
    return out


def _weighted_pair_count(per: Sequence[Tuple[np.ndarray, ...]],
                         w: np.ndarray, which: int) -> float:
    """sum over sites of w_k^T M_k w_k, with the self-pairs on each diagonal
    removed so a study resampled twice never pairs with itself.  Each unordered
    pair is counted twice, which cancels in the ratio."""
    tot = 0.0
    for blk in per:
        idx, M = blk[0], blk[which]
        wk = w[idx]
        tot += float(wk @ (M @ wk)) - float((np.diag(M) * wk * wk).sum())
    return tot
