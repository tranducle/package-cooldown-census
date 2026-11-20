"""Single-command reproducibility verifier for the five-ecosystem release-age cooldown census.

Modes:
  --mode verify : Runs all 14 unit and semantic-mutation tests under tests/,
                  regenerates figure-data CSV tables into a temporary directory from
                  results/census_summary.json, checks exact SHA-256 parity against
                  results/figure_data/*.csv, and verifies k=0 / no-native-feed invariants
                  and exact zero-sum Shapley additivity.
  --mode full   : Runs the complete archival census replay (requires locally downloaded
                  ecosyste.ms Parquet indices under COOLDOWN_DATA_ROOT or ./data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_verify() -> int:
    print("[1/3] Running constraint matcher and replay unit/mutation test suites...")
    r = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if r.returncode != 0:
        print(f"FAIL: unit test suite exited {r.returncode}")
        return r.returncode

    print("[2/3] Regenerating figure CSVs from results/census_summary.json and checking SHA-256 parity...")
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        env["COOLDOWN_SUMMARY_PATH"] = str(REPO_ROOT / "results" / "census_summary.json")
        env["COOLDOWN_FIGDATA_DIR"] = tmp
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "generate_figure_data.py")],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
        if r.returncode != 0:
            print("FAIL: scripts/generate_figure_data.py failed")
            return r.returncode

        expected_csvs = sorted((REPO_ROOT / "results" / "figure_data").glob("*.csv"))
        if not expected_csvs:
            print("FAIL: no reference CSVs found under results/figure_data/")
            return 1
        for ref_csv in expected_csvs:
            gen_csv = Path(tmp) / ref_csv.name
            if not gen_csv.exists():
                print(f"FAIL: regenerated CSV missing: {ref_csv.name}")
                return 1
            h_ref = sha256_file(ref_csv)
            h_gen = sha256_file(gen_csv)
            if h_ref != h_gen:
                print(f"FAIL: SHA-256 mismatch on {ref_csv.name}: ref={h_ref} gen={h_gen}")
                return 1
            print(f"  MATCH {ref_csv.name}: sha256={h_gen}")

    print("[3/3] Verifying deterministic k=0 / no-native-feed invariants and exact Shapley additivity...")
    summary = json.loads((REPO_ROOT / "results" / "census_summary.json").read_text(encoding="utf-8"))
    for eco in ("npm", "pypi", "cargo", "rubygems", "packagist"):
        g0 = summary["ecosystems"][eco]["gates"]["0"]
        assert g0["weighted_add_no_days_full"] == 0.0, f"{eco} k=0 add_no != 0"
        assert g0["unavail_fresh_pct_full"] == 0.0, f"{eco} k=0 fresh != 0"
        assert g0["unavail_sync_pct_full"] == 0.0, f"{eco} k=0 sync != 0"
    for eco in ("npm", "rubygems"):
        fg = summary["ecosystems"][eco]["gates"]["7"]["feed_gain_full_pp"]
        assert abs(fg) < 1e-12, f"{eco} no-native-feed invariant != 0"
    max_err = max(abs(v["additivity_err"]) for v in summary["shapley_pooled_ref"].values())
    assert max_err < 1e-9, f"Shapley additivity error too large: {max_err}"
    sum_delta = sum(v["total_gap"] for v in summary["shapley_pooled_ref"].values())
    assert abs(sum_delta) < 1e-6, f"Average-pairwise Shapley deltas do not sum to 0: {sum_delta}"
    assert summary.get("n_boot") == 5000, f"Unexpected bootstrap count: {summary.get('n_boot')}"
    print(f"  PASS: all k=0 and no-native-feed invariants == 0.0; n_boot=5000; sum(Delta)={sum_delta:.2e}; max Shapley additivity_err = {max_err:.2e}")
    print("ALL VERIFICATION CHECKS PASSED.")
    return 0


def run_full() -> int:
    data_root = Path(os.environ.get("COOLDOWN_DATA_ROOT", str(REPO_ROOT / "data")))
    if not (data_root / "ecosystems_open_data").exists():
        print(
            "ERROR: Full archival replay requires downloaded ecosyste.ms Parquet indices under "
            f"{data_root / 'ecosystems_open_data'}. See DATA.md for official download links."
        )
        return 2
    cmd = [sys.executable, str(REPO_ROOT / "experiments" / "run_census_replay.py"), "--mode", "official"]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or run the five-ecosystem cooldown census pipeline.")
    parser.add_argument("--mode", choices=("verify", "full"), default="verify")
    args = parser.parse_args()
    if args.mode == "verify":
        return run_verify()
    return run_full()


if __name__ == "__main__":
    raise SystemExit(main())
