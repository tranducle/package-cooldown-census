"""Pure vectorized continuous-time replay engine, analytical baselines, and exact 3-factor Shapley decomposition.

Zero file I/O; fully unit-tested on synthetic fixtures and k=0 negative controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

DAY = 86_400
GATES = (0, 1, 3, 7, 14, 30)
OPENNESS_CLASSES = ("exact", "patch", "minor", "wide")


@dataclass
class UpstreamData:
    ecosystem: str
    package_norm: str
    versions: List[str]
    t_pub: np.ndarray        # int64 seconds [V]
    rank: np.ndarray         # int64 distinct ranks [V], higher = newer version
    is_prerelease: np.ndarray  # bool [V]


def compute_eligibility(
    t_pub: np.ndarray,
    k_days: int,
    fixed_indices: Sequence[int] = (),
    advisory_sec: int | None = None,
    ideal_zero_lag: bool = False,
) -> np.ndarray:
    """Compute eligibility timestamps e(v) in integer seconds for each upstream version."""
    e = t_pub.astype(np.int64) + np.int64(k_days) * DAY
    if k_days == 0 or not fixed_indices:
        return e
    if ideal_zero_lag:
        for vi in fixed_indices:
            e[vi] = int(t_pub[vi])
    elif advisory_sec is not None:
        for vi in fixed_indices:
            pv = int(t_pub[vi])
            e[vi] = min(int(e[vi]), max(pv, int(advisory_sec)))
    return e


def build_step_function(
    rank: np.ndarray,
    admitted: np.ndarray,
    e: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build exact step function (T, val) where val[c, j] holds on [T[j], T[j+1]).

    admitted: bool [n_c, V]; e: eligibility timestamps [V]; rank: int64 [V].
    Returns (T [S], val [n_c, S]) with -1 meaning no version is eligible.
    """
    n_c, V = admitted.shape
    if V == 0:
        return np.array([0], dtype=np.int64), np.full((n_c, 1), -1, dtype=np.int64)
    order = np.argsort(e, kind="stable")
    e_sorted = e[order]
    rank_sorted = rank[order]
    adm_sorted = admitted[:, order]

    # Map distinct ranks back to original version index
    max_r = int(rank.max())
    idx_of_rank = np.full(max_r + 2, -1, dtype=np.int64)
    for vi in range(V):
        idx_of_rank[int(rank[vi])] = vi

    R = np.where(adm_sorted, rank_sorted[None, :], -1)
    P = np.maximum.accumulate(R, axis=1)

    # Collapse ties in e_sorted by keeping the LAST column for each unique eligibility timestamp
    uniq_T, first_idx = np.unique(e_sorted, return_index=True)
    # Last index for each unique timestamp is next_first_idx - 1 (and V - 1 for the last)
    last_idx = np.empty_like(first_idx)
    if len(first_idx) > 1:
        last_idx[:-1] = first_idx[1:] - 1
    last_idx[-1] = V - 1

    best_r = P[:, last_idx]
    val = np.where(best_r >= 0, idx_of_rank[np.maximum(best_r, 0)], -1)
    return uniq_T, val


def integrate_indicator(
    T: np.ndarray,
    val: np.ndarray,
    indicator: np.ndarray,
    c_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    include_unavail: bool = False,
) -> np.ndarray:
    """Exact continuous-time integral of indicator(val(c, t)) over [x, y) in integer seconds.

    indicator: bool [V]. If include_unavail=True, -1 maps to True (for unavailability integral);
    otherwise -1 maps to False (for vulnerable-version integral).
    """
    if len(c_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    # Append slot for index -1
    ind_ext = np.concatenate([indicator.astype(bool), [bool(include_unavail)]])
    # Map -1 -> len(indicator)
    V = len(indicator)
    mapped = np.where(val >= 0, val, V)
    I = ind_ext[mapped]  # [n_c, S]
    S = len(T)

    F = np.zeros((val.shape[0], S), dtype=np.int64)
    if S > 1:
        L = np.diff(T).astype(np.int64)
        F[:, 1:] = np.cumsum(I[:, :-1].astype(np.int64) * L[None, :], axis=1)

    def G(t: np.ndarray) -> np.ndarray:
        j = np.searchsorted(T, t, side="right") - 1
        out = np.zeros(len(t), dtype=np.int64)
        # Before T[0]: val is -1!
        before = j < 0
        if include_unavail and before.any():
            out[before] = t[before] - T[0]  # shift-invariant because G(y) - G(x) cancels T[0]!
        ok = ~before
        if ok.any():
            cj = c_idx[ok]
            jj = j[ok]
            dt_rem = (t[ok] - T[jj]).astype(np.int64)
            out[ok] = F[cj, jj] + I[cj, jj].astype(np.int64) * dt_rem
        return out

    y_eff = np.maximum(y, x)
    return G(y_eff) - G(x)


def value_at_time(T: np.ndarray, val: np.ndarray, c_idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate step function at exact timestamp t for each relation."""
    if len(c_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    j = np.searchsorted(T, t, side="right") - 1
    out = np.full(len(c_idx), -1, dtype=np.int64)
    ok = j >= 0
    if ok.any():
        out[ok] = val[c_idx[ok], j[ok]]
    return out


def unweighted_plugin_recovery(lags_sec: np.ndarray, k_days: int) -> float:
    """Analytical W10-style unweighted advisory-lag plug-in prediction of exemption recovery (%).

    For each advisory with lag = a_v - t_f (seconds), truncated delay under exemption is
    min(k, max(0, lag)), so predicted recovery fraction is 1 - min(k, max(0, lag)) / k.
    """
    if k_days <= 0 or len(lags_sec) == 0:
        return 0.0
    k_sec = float(k_days * DAY)
    eff_wait = np.minimum(k_sec, np.maximum(0.0, lags_sec.astype(np.float64)))
    return float(np.mean(1.0 - eff_wait / k_sec) * 100.0)


def exact_shapley_3factor(
    eval_fn,
) -> Dict[str, float]:
    """Exact, order-invariant 3-factor Shapley accounting decomposition over {S, G, C}.

    eval_fn(coalition: frozenset[str]) -> float evaluates the accounting metric when factors
    in `coalition` take the target ecosystem's values and factors outside take the reference
    distribution's values.

    Factors:
      - S: upstream native silent-fix lag distribution (a_OSV - t_f)
      - G: GHSA vendor curation lag (a_GHSA - a_OSV)
      - C: declared constraint openness & dependent-weight composition
    """
    factors = ("S", "G", "C")
    cache: Dict[frozenset[str], float] = {}
    for mask in range(8):
        coal = frozenset(factors[i] for i in range(3) if (mask >> i) & 1)
        cache[coal] = float(eval_fn(coal))

    weights = {0: 2.0 / 6.0, 1: 1.0 / 6.0, 2: 2.0 / 6.0}
    phi: Dict[str, float] = {}
    for f in factors:
        others = [x for x in factors if x != f]
        total = 0.0
        for mask in range(4):
            sub = frozenset(others[i] for i in range(2) if (mask >> i) & 1)
            w = weights[len(sub)]
            total += w * (cache[sub | {f}] - cache[sub])
        phi[f"phi_{f}"] = total

    total_gap = cache[frozenset(factors)] - cache[frozenset()]
    phi["total_gap"] = total_gap
    phi["reference_val"] = cache[frozenset()]
    phi["target_val"] = cache[frozenset(factors)]
    phi["additivity_err"] = abs((phi["phi_S"] + phi["phi_G"] + phi["phi_C"]) - total_gap)
    # Pairwise interaction terms (exact 2-way synergy relative to single-factor moves from empty set)
    f0 = cache[frozenset()]
    phi["interact_SG"] = cache[frozenset({"S", "G"})] - cache[frozenset({"S"})] - cache[frozenset({"G"})] + f0
    phi["interact_SC"] = cache[frozenset({"S", "C"})] - cache[frozenset({"S"})] - cache[frozenset({"C"})] + f0
    phi["interact_GC"] = cache[frozenset({"G", "C"})] - cache[frozenset({"G"})] - cache[frozenset({"C"})] + f0
    return phi


def shapley_cell_recovery_model(
    eco_cell_rates: np.ndarray,
    eco_cell_weights: np.ndarray,
    ref_cell_rates: np.ndarray,
    ref_cell_weights: np.ndarray,
    eco_native_rec: np.ndarray,
    eco_ghsa_loss: np.ndarray,
    ref_native_rec: np.ndarray,
    ref_ghsa_loss: np.ndarray,
) -> Dict[str, float]:
    """Evaluate exact 3-factor Shapley decomposition of 7-day GHSA recovery (%) on the 4 openness cells.

    For each constraint openness class c in {exact, patch, minor, wide}:
      - effective GHSA recovery in cell c = native_rec(c) - ghsa_loss(c)
      - total weighted recovery = sum_c (weight_c * base_add_c * (native_rec_c - ghsa_loss_c)) / sum_c (weight_c * base_add_c)
    Here `eco_cell_weights` is the normalized share of no-exemption added exposure C(P_k) contributed by
    constraint class c (so sum(w) = 1.0). Thus:
      - Factor C controls the constraint-openness exposure shares w_c.
      - Factor S controls the native-feed recovery profile native_rec_c (upstream silent-fix lag).
      - Factor G controls the GHSA vendor curation loss ghsa_loss_c = native_rec_c - ghsa_rec_c.
    Exact identity: f(empty) = ref GHSA recovery; f({S,G,C}) = eco GHSA recovery!
    """
    w_eco = eco_cell_weights / max(float(eco_cell_weights.sum()), 1e-12)
    w_ref = ref_cell_weights / max(float(ref_cell_weights.sum()), 1e-12)

    def eval_coal(coal: frozenset[str]) -> float:
        w = w_eco if "C" in coal else w_ref
        nat = eco_native_rec if "S" in coal else ref_native_rec
        loss = eco_ghsa_loss if "G" in coal else ref_ghsa_loss
        rec = np.clip(nat - loss, 0.0, 100.0)
        return float(np.sum(w * rec))

    return exact_shapley_3factor(eval_coal)


def clustered_bootstrap_ci(
    cluster_ids: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261119,
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Upstream-package clustered percentile bootstrap CI for ratio sum(num)/sum(den) * scale."""
    uniq_c, inv = np.unique(cluster_ids, return_inverse=True)
    n_cl = len(uniq_c)
    if n_cl == 0:
        return 0.0, 0.0, 0.0
    c_num = np.bincount(inv, weights=numerator.astype(np.float64), minlength=n_cl)
    c_den = np.bincount(inv, weights=denominator.astype(np.float64), minlength=n_cl)
    tot_den = float(c_den.sum())
    pt = (float(c_num.sum()) / tot_den * scale) if tot_den > 0 else 0.0
    if n_cl == 1:
        return pt, pt, pt
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_cl, size=(n_boot, n_cl))
    b_num = c_num[idx].sum(axis=1)
    b_den = c_den[idx].sum(axis=1)
    valid = b_den > 0
    if not valid.any():
        return pt, pt, pt
    vals = (b_num[valid] / b_den[valid]) * scale
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def clustered_bootstrap_gap_vs_unw_ci(
    cluster_ids: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    adv_lag_sec: np.ndarray,
    k_days: int,
    n_boot: int = 5000,
    seed: int = 20261119,
) -> Tuple[float, float, float]:
    """Paired upstream-clustered bootstrap CI for dependent-weighted Rec(k) minus unweighted B_unw(k) (pp)."""
    if k_days <= 0 or len(cluster_ids) == 0:
        return 0.0, 0.0, 0.0
    uniq_c, inv = np.unique(cluster_ids, return_inverse=True)
    n_cl = len(uniq_c)
    k_sec = float(k_days * DAY)
    adv_unw = 100.0 * (1.0 - np.clip(adv_lag_sec.astype(np.float64), 0.0, k_sec) / k_sec)
    c_num = np.bincount(inv, weights=numerator.astype(np.float64), minlength=n_cl)
    c_den = np.bincount(inv, weights=denominator.astype(np.float64), minlength=n_cl)
    c_unw_sum = np.bincount(inv, weights=adv_unw, minlength=n_cl)
    c_cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    tot_den = float(c_den.sum())
    pt_rec = (float(c_num.sum()) / tot_den * 100.0) if tot_den > 0 else 0.0
    pt_unw = float(c_unw_sum.sum() / max(float(c_cnt.sum()), 1.0))
    pt = pt_rec - pt_unw
    if n_cl == 1:
        return pt, pt, pt
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_cl, size=(n_boot, n_cl))
    b_num = c_num[idx].sum(axis=1)
    b_den = c_den[idx].sum(axis=1)
    b_unw = c_unw_sum[idx].sum(axis=1) / np.maximum(c_cnt[idx].sum(axis=1), 1.0)
    valid = b_den > 0
    if not valid.any():
        return pt, pt, pt
    vals = (b_num[valid] / b_den[valid]) * 100.0 - b_unw[valid]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(pt), float(lo), float(hi)
