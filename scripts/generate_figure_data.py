"""Generate SHA-256 verified figure-data CSV tables from results/census_summary.json."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
SUM_PATH = Path(os.environ.get("COOLDOWN_SUMMARY_PATH", str(PKG_ROOT / "results" / "census_summary.json")))
OUT_DIR = Path(os.environ.get("COOLDOWN_FIGDATA_DIR", str(PKG_ROOT / "results" / "figure_data")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

d = json.loads(SUM_PATH.read_text(encoding="utf-8"))
ECOS = ["npm", "pypi", "cargo", "rubygems", "packagist"]
GATES = [0, 1, 3, 7, 14, 30]
POS_GATES = [1, 3, 7, 14, 30]

written = []

# 1. malicious_detection_curves.csv: k_days vs empirical MAL-* blocking coverage (%)
p2a = OUT_DIR / "malicious_detection_curves.csv"
with p2a.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["k_days", "npm_prev_legit", "npm_live2025", "pypi_live2025", "pypi_prev_legit", "rubygems_live2025"])
    for k in GATES:
        sk = str(k)
        w.writerow([
            k,
            d["ecosystems"]["npm"]["mal_summary"]["blocking_prev_legit_pct"][sk],
            d["ecosystems"]["npm"]["mal_summary"]["blocking_live_2025_pct"][sk],
            d["ecosystems"]["pypi"]["mal_summary"]["blocking_live_2025_pct"][sk],
            d["ecosystems"]["pypi"]["mal_summary"]["blocking_prev_legit_pct"][sk],
            d["ecosystems"]["rubygems"]["mal_summary"]["blocking_live_2025_pct"][sk],
        ])
written.append(p2a)

# 2. exemption_recovery_curves.csv: k_days (1..30) vs GHSA recovery (%) for all 5 ecosystems
p2b = OUT_DIR / "exemption_recovery_curves.csv"
with p2b.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["k_days", "npm_ghsa", "pypi_ghsa", "cargo_ghsa", "cargo_osv", "rubygems_ghsa", "packagist_ghsa"])
    for k in POS_GATES:
        sk = str(k)
        w.writerow([
            k,
            d["ecosystems"]["npm"]["gates"][sk]["rec_ghsa_full_pct"],
            d["ecosystems"]["pypi"]["gates"][sk]["rec_ghsa_full_pct"],
            d["ecosystems"]["cargo"]["gates"][sk]["rec_ghsa_full_pct"],
            d["ecosystems"]["cargo"]["gates"][sk]["rec_osv_full_pct"],
            d["ecosystems"]["rubygems"]["gates"][sk]["rec_ghsa_full_pct"],
            d["ecosystems"]["packagist"]["gates"][sk]["rec_ghsa_full_pct"],
        ])
written.append(p2b)

# 3. advisory_feed_recovery_k7.csv: ordered bottom-to-top for horizontal dumbbell plot (Cargo top = y=5)
# Order top to bottom: Cargo (5), npm (4), RubyGems (3), Packagist (2), PyPI (1)
order_f3 = [("pypi", "PyPI", 1), ("packagist", "Packagist", 2), ("rubygems", "RubyGems", 3), ("npm", "npm", 4), ("cargo", "Cargo", 5)]
p3 = OUT_DIR / "advisory_feed_recovery_k7.csv"
with p3.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow([
        "y_idx", "eco_label", "b_unw", "rec_ghsa", "ghsa_err_lo", "ghsa_err_hi",
        "rec_osv", "osv_err_lo", "osv_err_hi", "rec_osv_eod", "feed_gain_pp", "gap_unw_pp", "annot_str",
    ])
    for eco, lbl, yidx in order_f3:
        g7 = d["ecosystems"][eco]["gates"]["7"]
        rg = g7["rec_ghsa_full_pct"]
        glo, ghi = g7["rec_ghsa_full_ci95"]
        ro = g7["rec_osv_full_pct"]
        olo, ohi = g7["rec_osv_full_ci95"]
        reod = g7["rec_osv_eod_full_pct"]
        bu = g7["baseline_unw_ghsa_pct"]
        fg = g7["feed_gain_full_pp"]
        gu = g7["gap_vs_unw_ghsa_pp"]
        annot = f"+{fg:.1f} pp feed | {gu:+.1f} pp vs B" if fg > 0 else f"0.0 pp feed | {gu:+.1f} pp vs B"
        w.writerow([
            yidx, lbl, bu, rg, round(rg - glo, 2), round(ghi - rg, 2),
            ro, round(ro - olo, 2), round(ohi - ro, 2), reod, fg, gu, annot,
        ])
written.append(p3)

# 4. resolution_unavailability.csv: k_days vs fresh & sync unavailability (%) for all 5 ecosystems
p4 = OUT_DIR / "resolution_unavailability.csv"
with p4.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow([
        "k_days",
        "npm_fresh", "pypi_fresh", "cargo_fresh", "rubygems_fresh", "packagist_fresh",
        "npm_sync", "pypi_sync", "cargo_sync", "rubygems_sync", "packagist_sync",
    ])
    for k in GATES:
        sk = str(k)
        w.writerow([
            k,
            *[d["ecosystems"][e]["gates"][sk]["unavail_fresh_pct_full"] for e in ECOS],
            *[d["ecosystems"][e]["gates"][sk]["unavail_sync_pct_full"] for e in ECOS],
        ])
written.append(p4)

# 5. shapley_decomposition_k7.csv: exact 3-factor Shapley shares relative to 5-ecosystem mean (51.58%)
from decimal import Decimal, ROUND_HALF_UP


def _r2(val: float) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


order_f5 = [("packagist", "Packagist", 1), ("pypi", "PyPI", 2), ("rubygems", "RubyGems", 3), ("npm", "npm", 4), ("cargo", "Cargo", 5)]
p5 = OUT_DIR / "shapley_decomposition_k7.csv"
with p5.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["y_idx", "eco_label", "ref_val", "target_val", "total_gap", "phi_S", "phi_G", "phi_C", "annot_str"])
    for eco, lbl, yidx in order_f5:
        sh = d["shapley_pooled_ref"][eco]
        annot = f"{sh['total_gap']:+.1f} pp ({sh['target_val']:.1f}\\%)"
        w.writerow([
            yidx, lbl, _r2(sh["reference_val"]), _r2(sh["target_val"]),
            _r2(sh["total_gap"]), _r2(sh["phi_S"]), _r2(sh["phi_G"]), _r2(sh["phi_C"]), annot,
        ])
written.append(p5)

# Write SHA-256 manifest for figure_data
manifest = {
    "source_summary": "results/census_summary.json",
    "source_summary_sha256": hashlib.sha256(SUM_PATH.read_bytes()).hexdigest(),
    "files": {
        p.name: {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in written
    },
}
mp = OUT_DIR / "figure_data_manifest.json"
mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Generated figure_data CSVs + manifest:", list(manifest["files"].keys()))
