"""Unit tests verifying the continuous-time cooldown replay engine, shift invariance, and exact Shapley accounting."""

from __future__ import annotations

import unittest
import numpy as np

import sys
from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src import cooldown_replay_engine as eng


class TestCooldownReplayEngine(unittest.TestCase):
    def test_negative_control_k0_exact_zero(self):
        # Upstream with vulnerable v0 (t=0) and fixed v1 (t=10d)
        t_pub = np.array([0, 10 * eng.DAY], dtype=np.int64)
        rank = np.array([0, 1], dtype=np.int64)
        admitted = np.array([[True, True], [True, False]], dtype=bool)
        vuln = np.array([True, False], dtype=bool)

        e0 = eng.compute_eligibility(t_pub, k_days=0)
        T0, val0 = eng.build_step_function(rank, admitted, e0)

        c_idx = np.array([0, 1], dtype=np.int64)
        x = np.array([10 * eng.DAY, 10 * eng.DAY], dtype=np.int64)
        y = np.array([40 * eng.DAY, 40 * eng.DAY], dtype=np.int64)
        exp0 = eng.integrate_indicator(T0, val0, vuln, c_idx, x, y, include_unavail=False)
        # Floating relation (c=0) gets fixed immediately at t=10d -> 0 exposure; pinned (c=1) stays vulnerable 30d
        self.assertEqual(exp0.tolist(), [0, 30 * eng.DAY])

        # Fresh unavailability at t_d=10d under k=0 must be identically 0
        v_at_td = eng.value_at_time(T0, val0, c_idx, x)
        self.assertTrue(np.all(v_at_td >= 0))

    def test_cooldown_and_advisory_exemption_exact_days(self):
        # Vulnerable v0 at day 0, fixed v1 at day 10
        # Under k=7d no exemption: v1 eligible at day 17 -> floating relation exposed 7 full days
        # Under GHSA advisory at day 15: v1 eligible at day 15 -> exposed 5 days (2/7 = 28.57% recovery)
        # Under native OSV advisory at day 11: v1 eligible at day 11 -> exposed 1 day (6/7 = 85.71% recovery)
        # Under idealized zero-lag exemption: v1 eligible at day 10 -> exposed 0 days (100% recovery)
        t_pub = np.array([0, 10 * eng.DAY], dtype=np.int64)
        rank = np.array([0, 1], dtype=np.int64)
        admitted = np.array([[True, True]], dtype=bool)
        vuln = np.array([True, False], dtype=bool)
        c_idx = np.array([0], dtype=np.int64)
        x = np.array([10 * eng.DAY], dtype=np.int64)
        y = np.array([40 * eng.DAY], dtype=np.int64)

        T_no, V_no = eng.build_step_function(rank, admitted, eng.compute_eligibility(t_pub, 7))
        T_ghsa, V_ghsa = eng.build_step_function(rank, admitted, eng.compute_eligibility(t_pub, 7, [1], 15 * eng.DAY))
        T_osv, V_osv = eng.build_step_function(rank, admitted, eng.compute_eligibility(t_pub, 7, [1], 11 * eng.DAY))
        T_zero, V_zero = eng.build_step_function(rank, admitted, eng.compute_eligibility(t_pub, 7, [1], ideal_zero_lag=True))

        e_no = eng.integrate_indicator(T_no, V_no, vuln, c_idx, x, y)[0]
        e_ghsa = eng.integrate_indicator(T_ghsa, V_ghsa, vuln, c_idx, x, y)[0]
        e_osv = eng.integrate_indicator(T_osv, V_osv, vuln, c_idx, x, y)[0]
        e_zero = eng.integrate_indicator(T_zero, V_zero, vuln, c_idx, x, y)[0]

        self.assertEqual(e_no, 7 * eng.DAY)
        self.assertEqual(e_ghsa, 5 * eng.DAY)
        self.assertEqual(e_osv, 1 * eng.DAY)
        self.assertEqual(e_zero, 0)

    def test_fresh_vs_synchronized_unavailability(self):
        # Dependent published at day 12 requires >= v1 (where v1 published at day 10)
        # Under k=7d: v1 eligible at day 17.
        # Fresh-install at t_d=12d: unresolvable (-1) for 5 days [12d, 17d)!
        # Synchronized-ecosystem starting at t_d + 7d = 19d: ALREADY resolvable (0 unavailable seconds)!
        t_pub = np.array([0, 10 * eng.DAY], dtype=np.int64)
        rank = np.array([0, 1], dtype=np.int64)
        admitted = np.array([[False, True]], dtype=bool)
        dummy_ind = np.array([False, False], dtype=bool)
        c_idx = np.array([0], dtype=np.int64)

        T7, V7 = eng.build_step_function(rank, admitted, eng.compute_eligibility(t_pub, 7))
        # Fresh at day 12 -> -1
        self.assertEqual(eng.value_at_time(T7, V7, c_idx, np.array([12 * eng.DAY]))[0], -1)
        u_fresh_sec = eng.integrate_indicator(
            T7, V7, dummy_ind, c_idx, np.array([12 * eng.DAY]), np.array([30 * eng.DAY]), include_unavail=True
        )[0]
        self.assertEqual(u_fresh_sec, 5 * eng.DAY)
        # Synchronized on [12d + 7d, 30d) = [19d, 30d) -> 0 unavailable seconds!
        u_sync_sec = eng.integrate_indicator(
            T7, V7, dummy_ind, c_idx, np.array([19 * eng.DAY]), np.array([30 * eng.DAY]), include_unavail=True
        )[0]
        self.assertEqual(u_sync_sec, 0)

    def test_shapley_exact_additivity_and_self_zero(self):
        w_ref = np.array([0.0, 0.2, 0.5, 0.3])
        nat_ref = np.array([0.0, 60.0, 70.0, 80.0])
        loss_ref = np.array([0.0, 5.0, 10.0, 15.0])

        # Self comparison must be identically 0.0
        self_res = eng.shapley_cell_recovery_model(
            nat_ref, w_ref, nat_ref, w_ref, nat_ref, loss_ref, nat_ref, loss_ref
        )
        self.assertAlmostEqual(self_res["total_gap"], 0.0, places=12)
        self.assertAlmostEqual(self_res["phi_S"], 0.0, places=12)
        self.assertAlmostEqual(self_res["phi_G"], 0.0, places=12)
        self.assertAlmostEqual(self_res["phi_C"], 0.0, places=12)

        # Pure GHSA vendor curation lag shift (+20 pp extra loss uniformly) -> phi_G must be exactly -20.0!
        loss_eco = loss_ref + 20.0
        pure_g = eng.shapley_cell_recovery_model(
            nat_ref, w_ref, nat_ref, w_ref, nat_ref, loss_eco, nat_ref, loss_ref
        )
        self.assertAlmostEqual(pure_g["phi_G"], -20.0, places=10)
        self.assertAlmostEqual(pure_g["phi_S"], 0.0, places=10)
        self.assertAlmostEqual(pure_g["phi_C"], 0.0, places=10)
        self.assertLess(pure_g["additivity_err"], 1e-12)

        # General mixed shift -> exact additivity holds
        w_eco = np.array([0.0, 0.1, 0.2, 0.7])
        nat_eco = np.array([0.0, 40.0, 55.0, 65.0])
        mixed = eng.shapley_cell_recovery_model(
            nat_eco, w_eco, nat_ref, w_ref, nat_eco, loss_eco, nat_ref, loss_ref
        )
        self.assertLess(mixed["additivity_err"], 1e-12)
        self.assertAlmostEqual(
            mixed["phi_S"] + mixed["phi_G"] + mixed["phi_C"],
            mixed["total_gap"],
            places=12,
        )

    def test_paired_gap_vs_unw_bootstrap_and_crn_stability(self):
        """Verify paired gap bootstrap intervals and common-random-numbers (CRN) reproducibility."""
        cl = np.array(["a", "a", "b", "c", "d", "e"])
        num = np.array([10.0, 20.0, 30.0, 5.0, 15.0, 25.0])
        den = np.array([20.0, 30.0, 40.0, 10.0, 25.0, 30.0])
        lags = np.array([0, 2, 10, 5, 1, 7]) * eng.DAY
        # CRN: identical inputs + same seed yield bit-identical CIs
        c1 = eng.clustered_bootstrap_ci(cl, num, den, n_boot=500, seed=20261126, scale=100.0)
        c2 = eng.clustered_bootstrap_ci(cl, num, den, n_boot=500, seed=20261126, scale=100.0)
        self.assertEqual(c1, c2)
        # Paired gap vs unweighted baseline
        pt, lo, hi = eng.clustered_bootstrap_gap_vs_unw_ci(cl, num, den, lags, k_days=7, n_boot=500, seed=20261126)
        b_unw = eng.unweighted_plugin_recovery(lags, 7)
        self.assertAlmostEqual(pt, c1[0] - b_unw, places=10)
        self.assertLessEqual(lo, pt)
        self.assertLessEqual(pt, hi)


if __name__ == "__main__":
    unittest.main()
