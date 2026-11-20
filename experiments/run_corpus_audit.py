"""Verify Parquet index manifests, advisory corpus integrity, and empirical constraint parse coverage across 5,000 sampled constraints."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from packaging.specifiers import SpecifierSet
from packaging.version import Version as Pep440

import sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src import ecosystem_matchers as er

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("COOLDOWN_DATA_ROOT", str(REPO_ROOT / "data")))
NPM_IDX = BASE / "ecosystems_open_data" / "index"
MULTI_IDX = BASE / "ecosystems_open_data" / "index_multieco"
DATA_DIR = BASE
ART_DIR = Path(os.environ.get("COOLDOWN_RESULTS_DIR", str(REPO_ROOT / "results")))
ECOSYSTEMS = ("npm", "pypi", "cargo", "rubygems", "packagist")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifests": {},
        "advisory_corpus": {},
        "constraint_parse_audit": {},
    }

    # 1. Verify multieco manifests
    for tbl in ("packages", "versions", "dependencies"):
        mp = MULTI_IDX / f"{tbl}.MANIFEST.json"
        if not mp.exists():
            raise FileNotFoundError(f"Missing manifest: {mp}")
        md = json.loads(mp.read_text(encoding="utf-8"))
        report["manifests"][f"multieco_{tbl}"] = {
            "path": str(mp),
            "sha256": sha256_file(mp),
            "copy_terminated": md.get("copy_terminated"),
            "rows_seen": md.get("rows_seen"),
            "counts_by_eco": md.get("counts_by_eco") or md.get("rows_kept_by_eco"),
            "elapsed_sec": md.get("elapsed_sec"),
        }

    # Check npm index file counts
    for tbl in ("packages", "versions", "dependencies"):
        fs = sorted((NPM_IDX / tbl).glob("*.parquet"))
        report["manifests"][f"npm_{tbl}"] = {
            "dir": str(NPM_IDX / tbl),
            "n_parts": len(fs),
        }

    # 2. Audit advisory corpus and export temporal split tables
    adv_path = DATA_DIR / "advisories_census.parquet"
    mal_path = DATA_DIR / "malicious_packages.parquet"
    adv_df = pl.read_parquet(adv_path)
    mal_df = pl.read_parquet(mal_path)

    g_adv = adv_df.group_by("ecosystem").len().sort("ecosystem")
    g_mal = mal_df.group_by("ecosystem").len().sort("ecosystem")
    report["advisory_corpus"] = {
        "advisories_parquet": str(adv_path),
        "advisories_sha256": sha256_file(adv_path),
        "advisories_rows": adv_df.height,
        "advisories_by_eco": dict(zip(g_adv["ecosystem"].to_list(), g_adv["len"].to_list())),
        "mal_parquet": str(mal_path),
        "mal_sha256": sha256_file(mal_path),
        "mal_rows": mal_df.height,
        "mal_by_eco": dict(zip(g_mal["ecosystem"].to_list(), g_mal["len"].to_list())),
        "duplicate_eco_ghsa_pkg": int(adv_df.height - adv_df.unique(subset=["ecosystem", "ghsa_id", "package_norm"]).height),
        "null_t_ghsa": int(adv_df["t_ghsa_sec"].null_count()),
        "null_t_osv": int(adv_df["t_osv_earliest_sec"].null_count()),
    }

    # Export temporal split CSVs (pre-2022 vs 2022 to 2026 by GHSA publication date)
    cutoff_2022 = int(datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    early_csv = ART_DIR / "audit_adv_pre2022.csv"
    recent_csv = ART_DIR / "audit_adv_2022_2026.csv"
    cols = ["record_id", "ecosystem", "package_key", "event_time", "severity", "has_vendor_lag"]
    with early_csv.open("w", newline="", encoding="utf-8") as f1, recent_csv.open("w", newline="", encoding="utf-8") as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(cols)
        w2.writerow(cols)
        for row in adv_df.iter_rows(named=True):
            t_g = int(row["t_ghsa_sec"])
            iso_t = datetime.fromtimestamp(t_g, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec = [
                f"{row['ecosystem']}:{row['ghsa_id']}:{row['package_norm']}",
                row["ecosystem"],
                f"{row['ecosystem']}:{row['package_norm']}",
                iso_t,
                row["severity"] or "UNKNOWN",
                "lag_gt0" if int(row["vendor_lag_sec"]) > 0 else "lag_0",
            ]
            if t_g < cutoff_2022:
                w1.writerow(rec)
            else:
                w2.writerow(rec)

    # 3. Sample 1,000 real dependency requirement strings per ecosystem to measure live parse coverage!
    rng = random.Random(20261119)
    pypi_diff_vers = [
        "0.5.0", "1.0.0", "1.2.0", "1.2.3", "1.5.0", "2.0.0", "2.1.0", "3.0.0", "1.3.0a1"
    ]
    pypi_pvs = [er.parse_version("pypi", v) for v in pypi_diff_vers]

    for eco in ECOSYSTEMS:
        if eco == "npm":
            d_files = sorted((NPM_IDX / "dependencies").glob("*.parquet"))
        else:
            d_files = sorted((MULTI_IDX / "dependencies" / eco).glob("*.parquet"))
        chosen_files = rng.sample(d_files, min(20, len(d_files)))
        reqs: list[str] = []
        for fp in chosen_files:
            df = pl.read_parquet(fp, columns=["requirements"])
            vals = df["requirements"].to_list()
            if len(vals) > 100:
                vals = rng.sample(vals, 100)
            reqs.extend("" if v is None else str(v) for v in vals)
        if len(reqs) > 1000:
            reqs = rng.sample(reqs, 1000)

        parsed_ok = 0
        fam_counts = Counter()
        bad_examples = Counter()
        pypi_diff_checked = 0
        pypi_diff_matched = 0

        for rq in reqs:
            fn, fam = er.compile_constraint(eco, rq)
            if fn is not None:
                parsed_ok += 1
                fam_counts[fam] += 1
                if eco == "pypi":
                    # Differential verification against PyPA packaging.specifiers!
                    s_clean = rq.strip()
                    if ";" in s_clean:
                        s_clean = s_clean.split(";", 1)[0].strip()
                    if s_clean.startswith("(") and s_clean.endswith(")"):
                        s_clean = s_clean[1:-1].strip()
                    if s_clean in ("", "*"):
                        s_clean = ""
                    spec = SpecifierSet(s_clean, prereleases=False)
                    ref_mask = [
                        (not (Pep440(v).is_prerelease or Pep440(v).is_devrelease))
                        and spec.contains(Pep440(v), prereleases=False)
                        for v in pypi_diff_vers
                    ]
                    got_mask = fn(pypi_pvs)
                    pypi_diff_checked += 1
                    if got_mask == ref_mask:
                        pypi_diff_matched += 1
            else:
                bad_examples[rq] += 1

        entry = {
            "n_sampled": len(reqs),
            "parsed_ok": parsed_ok,
            "parse_rate_pct": round(100.0 * parsed_ok / max(len(reqs), 1), 2),
            "openness_dist": dict(fam_counts),
            "top_unparseable": bad_examples.most_common(5),
        }
        if eco == "pypi":
            entry["pypi_differential_checked"] = pypi_diff_checked
            entry["pypi_differential_matched"] = pypi_diff_matched
        report["constraint_parse_audit"][eco] = entry
        print(f"  [audit {eco}] parsed={parsed_ok}/{len(reqs)} ({entry['parse_rate_pct']}%), openness={dict(fam_counts)}", flush=True)

    out_p = ART_DIR / "corpus_audit_metrics.json"
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
