# 100-Record Stratified Manual Linkage and Monotonicity Audit

- **Audit Date:** 2026-11-19
- **Source Table:** `results/manual_audit_sample_100.csv`
- **Sample Design:** 100 evaluated advisories (20 randomly sampled per ecosystem from the complete 100% census of 9,887 evaluated advisories; deterministic seed `20261119 + len(ecosystem)`).
- **Monotonicity Violations (`add_osv_7 <= add_ghsa_7 <= add_no_7`):** `0 / 100` (`0.0%`; two-sided 95% Clopper-Pearson upper bound `3.6%` pooled, `16.8%` per 20-record stratum).
- **Mechanical Linkage or Timestamp Extraction Errors:** `0 / 100`.
- **Empirical Clock Distribution Within Sample:**
  - GHSA published on or before fix (`lag_ghsa_days <= 0`): `7 / 100`
  - Earliest native OSV published on or before fix (`lag_osv_days <= 0`): `15 / 100`
  - Strictly positive GHSA-versus-native vendor lag (`vendor_lag_days > 0`): `29 / 100`
  - Pre-2022 multi-year GHSA backfills (`vendor_lag_days > 365`): `3 / 100`:
    1. `pypi`: `PYSEC-2020-9` on `ansible` (fix 2020; vendor lag `407.02` days)
    2. `cargo`: `RUSTSEC-2020-0015` on `openssl-src` (fix 2020; vendor lag `487.36` days)
    3. `cargo`: `RUSTSEC-2019-0008` on `simd-json` (fix 2019; vendor lag `793.36` days)
- **Summary:** All 100 audited records match upstream JSON advisory records and satisfy counterfactual policy monotonicity without extraction errors.
