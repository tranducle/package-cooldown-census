"""Build unified multi-ecosystem advisory catalog (GHSA + native OSV PYSEC/RUSTSEC/DRUPAL + MAL-*).

Reads local SHA-256 verified GHSA and OSV archives for npm, PyPI, Cargo, RubyGems, and Packagist.
Links GHSA reviewed advisories with native OSV records via CVE/alias graphs and (pkg, fixed_version)
to measure exact t_ghsa_sec vs t_osv_earliest_sec cleanly. Also extracts MAL-* malicious-package
records with OSV publication timestamps.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.ecosystem_matchers import normalize_package_name

_PACK_ROOT = Path(__file__).resolve().parent.parent
ADV_DIR = Path(os.environ.get("COOLDOWN_ADV_DIR", str(_PACK_ROOT / "data" / "advisories")))
OUT_DIR = Path(os.environ.get("COOLDOWN_DATA_DIR", str(_PACK_ROOT / "data")))

GHSA_ZIP = ADV_DIR / "github_advisory_database_674392f1b2bbbe4ec5f1da49be6fa9a7fdb89b22.zip"
OSV_ZIPS = {
    "npm": ADV_DIR / "osv_npm_all.zip",
    "pypi": ADV_DIR / "osv_pypi_all.zip",
    "cargo": ADV_DIR / "osv_cargo_all.zip",
    "rubygems": ADV_DIR / "osv_rubygems_all.zip",
    "packagist": ADV_DIR / "osv_packagist_all.zip",
}
GHSA_ECO_MAP = {
    "npm": "npm",
    "PyPI": "pypi",
    "crates.io": "cargo",
    "RubyGems": "rubygems",
    "Packagist": "packagist",
}
CUTOFF_SEC = int(datetime(2026, 2, 5, 0, 0, 0, tzinfo=timezone.utc).timestamp())
MIN_SEC = int(datetime(2015, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def parse_iso_sec(val: str | None) -> int | None:
    if not val or not isinstance(val, str):
        return None
    s = val.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp())
    except ValueError:
        return None


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Index ALL native OSV vulnerability advisories (non-MAL, non-GHSA) + extract ALL MAL-* records!
    # Map: (eco, alias_id) -> min(t_pub, native_id) AND (eco, pkg_norm, fixed_ver) -> min(t_pub, native_id)
    native_by_alias: dict[tuple[str, str], tuple[int, str]] = {}
    native_by_pkg_fix: dict[tuple[str, str, str], tuple[int, str]] = {}
    mal_rows: list[dict] = []
    osv_stats: dict[str, dict] = {}

    for eco, zpath in OSV_ZIPS.items():
        st = Counter()
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                d = json.loads(zf.read(name))
                if d.get("withdrawn"):
                    st["withdrawn"] += 1
                    continue
                oid = str(d.get("id") or "")
                t_pub = parse_iso_sec(d.get("published"))
                if t_pub is None:
                    st["no_pub_ts"] += 1
                    continue

                if oid.startswith("MAL-"):
                    if t_pub > CUTOFF_SEC:
                        st["mal_after_cutoff"] += 1
                        continue
                    for aff in d.get("affected") or []:
                        pkg_obj = aff.get("package") or {}
                        pname = pkg_obj.get("name")
                        if not pname:
                            continue
                        pnorm = normalize_package_name(eco, str(pname))
                        vers = [str(v) for v in (aff.get("versions") or []) if v]
                        # Also check introduced == "0" (whole package malicious)
                        whole_pkg = False
                        for r in aff.get("ranges") or []:
                            evs = r.get("events") or []
                            if any(ev.get("introduced") == "0" for ev in evs) and not any("fixed" in ev for ev in evs):
                                whole_pkg = True
                        mal_rows.append(
                            {
                                "ecosystem": eco,
                                "mal_id": oid,
                                "package_raw": str(pname),
                                "package_norm": pnorm,
                                "t_osv_pub_sec": t_pub,
                                "versions_json": json.dumps(vers),
                                "whole_package": whole_pkg,
                            }
                        )
                        st["mal_kept"] += 1
                    continue

                # Native non-GHSA advisories (PYSEC, RUSTSEC, DRUPAL, OSV, etc.)
                if not oid.startswith("GHSA-"):
                    st["native_vuln"] += 1
                    aliases = [oid] + [str(a) for a in (d.get("aliases") or []) if a]
                    for al in aliases:
                        key = (eco, al)
                        prev = native_by_alias.get(key)
                        if prev is None or t_pub < prev[0]:
                            native_by_alias[key] = (t_pub, oid)
                    for aff in d.get("affected") or []:
                        pkg_obj = aff.get("package") or {}
                        pname = pkg_obj.get("name")
                        if not pname:
                            continue
                        pnorm = normalize_package_name(eco, str(pname))
                        for r in aff.get("ranges") or []:
                            for ev in r.get("events") or []:
                                fx = ev.get("fixed")
                                if fx:
                                    kf = (eco, pnorm, str(fx))
                                    prev = native_by_pkg_fix.get(kf)
                                    if prev is None or t_pub < prev[0]:
                                        native_by_pkg_fix[kf] = (t_pub, oid)
        osv_stats[eco] = dict(st)
        print(f"[OSV {eco}] {dict(st)}", flush=True)

    # 2. Parse ALL reviewed GHSA advisories across our 5 ecosystems & join to native OSV timestamps!
    ghsa_rows: list[dict] = []
    ghsa_stats = {e: Counter() for e in OSV_ZIPS}

    with zipfile.ZipFile(GHSA_ZIP) as zf:
        for name in zf.namelist():
            if "/advisories/github-reviewed/" not in name or not name.endswith(".json"):
                continue
            d = json.loads(zf.read(name))
            if d.get("withdrawn"):
                continue
            gid = str(d.get("id") or "")
            t_ghsa = parse_iso_sec(d.get("published"))
            if t_ghsa is None:
                continue
            db_spec = d.get("database_specific") or {}
            severity = str(db_spec.get("severity") or "UNKNOWN").upper()
            cwes = db_spec.get("cwe_ids") or []
            aliases = [gid] + [str(a) for a in (d.get("aliases") or []) if a]
            cves = [a for a in aliases if a.startswith("CVE-")]

            # Group affected blocks per (eco, pkg_norm) within one GHSA advisory
            grouped: dict[tuple[str, str], dict] = {}
            for aff in d.get("affected") or []:
                pkg_obj = aff.get("package") or {}
                eco_raw = pkg_obj.get("ecosystem")
                eco = GHSA_ECO_MAP.get(str(eco_raw))
                if eco is None:
                    continue
                pname = pkg_obj.get("name")
                if not pname:
                    continue
                pnorm = normalize_package_name(eco, str(pname))
                ranges = aff.get("ranges") or []
                fixed_list = []
                range_events = []
                for r in ranges:
                    evs = r.get("events") or []
                    range_events.append(evs)
                    for ev in evs:
                        fx = ev.get("fixed")
                        if fx:
                            fixed_list.append(str(fx))
                if not fixed_list:
                    ghsa_stats[eco]["no_fixed_version"] += 1
                    continue
                gkey = (eco, pnorm)
                if gkey not in grouped:
                    grouped[gkey] = {
                        "package_raw": str(pname),
                        "fixed": [],
                        "ranges": [],
                    }
                grouped[gkey]["fixed"].extend(fixed_list)
                grouped[gkey]["ranges"].extend(range_events)

            for (eco, pnorm), info in grouped.items():
                fixed_unique = sorted(set(info["fixed"]))
                # Look up earliest native OSV timestamp via aliases OR (eco, pkg_norm, fixed_ver)
                best_native: tuple[int, str] | None = None
                match_method = "none"
                for al in aliases:
                    cand = native_by_alias.get((eco, al))
                    if cand is not None and (best_native is None or cand[0] < best_native[0]):
                        best_native = cand
                        match_method = "alias"
                if best_native is None:
                    for fx in fixed_unique:
                        cand = native_by_pkg_fix.get((eco, pnorm, fx))
                        if cand is not None and (best_native is None or cand[0] < best_native[0]):
                            best_native = cand
                            match_method = "pkg_fix"

                t_native = best_native[0] if best_native is not None else None
                native_id = best_native[1] if best_native is not None else None
                t_earliest = min(t_ghsa, t_native) if t_native is not None else t_ghsa
                vendor_lag_sec = t_ghsa - t_earliest

                ghsa_rows.append(
                    {
                        "ecosystem": eco,
                        "ghsa_id": gid,
                        "package_raw": info["package_raw"],
                        "package_norm": pnorm,
                        "severity": severity,
                        "is_cwe506": "CWE-506" in cwes,
                        "cve_ids_json": json.dumps(cves),
                        "fixed_versions_json": json.dumps(fixed_unique),
                        "ranges_json": json.dumps(info["ranges"]),
                        "t_ghsa_sec": t_ghsa,
                        "t_native_sec": t_native if t_native is not None else -1,
                        "native_id": native_id or "",
                        "native_match_method": match_method,
                        "t_osv_earliest_sec": t_earliest,
                        "vendor_lag_sec": vendor_lag_sec,
                    }
                )
                ghsa_stats[eco]["kept_fixed_advisories"] += 1
                if t_native is not None:
                    ghsa_stats[eco]["matched_native"] += 1
                    if vendor_lag_sec > 0:
                        ghsa_stats[eco]["native_earlier_than_ghsa"] += 1

    adv_df = pl.DataFrame(ghsa_rows)
    mal_df = pl.DataFrame(mal_rows)
    adv_path = OUT_DIR / "advisories_census.parquet"
    mal_path = OUT_DIR / "malicious_packages.parquet"
    adv_df.write_parquet(adv_path, compression="zstd")
    mal_df.write_parquet(mal_path, compression="zstd")

    receipt = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "advisories_parquet": str(adv_path),
        "advisories_sha256": hashlib.sha256(adv_path.read_bytes()).hexdigest(),
        "advisories_rows": adv_df.height,
        "mal_parquet": str(mal_path),
        "mal_sha256": hashlib.sha256(mal_path.read_bytes()).hexdigest(),
        "mal_rows": mal_df.height,
        "osv_stats": osv_stats,
        "ghsa_stats": {e: dict(c) for e, c in ghsa_stats.items()},
    }
    rec_path = OUT_DIR / "advisory_build_receipt.json"
    rec_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n=== ADVISORY CATALOG BUILD SUMMARY ===")
    for eco in OSV_ZIPS:
        print(f"  {eco}: {dict(ghsa_stats[eco])}")
    print(f"Total fixed advisories={adv_df.height:,}, MAL rows={mal_df.height:,} in {receipt['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
