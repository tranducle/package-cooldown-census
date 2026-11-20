"""Five-Ecosystem Release-Age Cooldown Replay, MAL-* Detection Curves, Fan-Out, and Shapley Runner.

Supports:
  --mode pilot     : Bounded 100-advisory-per-ecosystem pilot check.
  --mode official  : Full 100% census run (cap=0, w_h=1.0 across all eligible advisories,
                     active stratum, MAL-* two-sided frontier, mirror-filtered 50-advisory/eco fan-out check,
                     100-record manual audit export, top-3 hub deletion, 2022 to 2026 recent-era check,
                     exact average-pairwise 3-factor Shapley decomposition, and 5,000 common-random-number
                     upstream-clustered bootstrap intervals with seed 20261119).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import polars as pl

from src import ecosystem_matchers as er
from src import cooldown_replay_engine as eng

ECOSYSTEMS = ("npm", "pypi", "cargo", "rubygems", "packagist")
PKG_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("COOLDOWN_DATA_ROOT", str(PKG_ROOT / "data")))
NPM_IDX = DATA_ROOT / "ecosystems_open_data" / "index"
MULTI_IDX = DATA_ROOT / "ecosystems_open_data" / "index_multieco"
RESULTS_DIR = Path(os.environ.get("COOLDOWN_RESULTS_DIR", str(PKG_ROOT / "results")))
ART_DIR = RESULTS_DIR

CUTOFF_SEC = int(datetime(2026, 2, 5, 0, 0, 0, tzinfo=timezone.utc).timestamp())
MIN_VER_SEC = int(datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
FIX_START_SEC = int(datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
RECENT_START_SEC = int(datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
MAL_LIVE_2024_SEC = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
MAL_LIVE_2025_SEC = int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
SEED = 20261119


def get_parquet_paths(eco: str) -> tuple[list[Path], list[Path], list[Path]]:
    if eco == "npm":
        p_files = sorted((NPM_IDX / "packages").glob("*.parquet"))
        v_files = sorted((NPM_IDX / "versions").glob("*.parquet"))
        d_files = sorted((NPM_IDX / "dependencies").glob("*.parquet"))
    else:
        p_files = sorted((MULTI_IDX / "packages" / eco).glob("*.parquet"))
        v_files = sorted((MULTI_IDX / "versions" / eco).glob("*.parquet"))
        d_files = sorted((MULTI_IDX / "dependencies" / eco).glob("*.parquet"))
    return p_files, v_files, d_files


def vuln_mask_for_ranges(
    eco: str,
    pvs: list[er.ParsedVersion],
    ranges_list: list[list[dict]],
) -> np.ndarray:
    """Compute boolean mask [V] of non-prerelease versions affected by GHSA/OSV range events.

    Follows OSV schema: an events array may contain multiple (introduced, fixed/last_affected)
    segments for multi-branch backports; each segment is evaluated independently and unioned.
    """
    V = len(pvs)
    mask = np.zeros(V, dtype=bool)
    for evs in ranges_list:
        segments: list[tuple[bool, er.ParsedVersion | None, er.ParsedVersion | None, er.ParsedVersion | None]] = []
        cur_zero = False
        cur_intro: er.ParsedVersion | None = None
        cur_fixed: er.ParsedVersion | None = None
        cur_last: er.ParsedVersion | None = None
        has_open = False

        def flush() -> None:
            nonlocal cur_zero, cur_intro, cur_fixed, cur_last, has_open
            if has_open and (cur_zero or cur_intro is not None):
                segments.append((cur_zero, cur_intro, cur_fixed, cur_last))
            cur_zero = False
            cur_intro = None
            cur_fixed = None
            cur_last = None
            has_open = False

        for ev in evs:
            if "introduced" in ev and ev["introduced"] is not None:
                if has_open:
                    flush()
                raw_i = str(ev["introduced"]).strip()
                if raw_i == "0":
                    cur_zero = True
                else:
                    cur_intro = er.parse_version(eco, raw_i)
                has_open = True
            elif "fixed" in ev and ev["fixed"] is not None:
                cur_fixed = er.parse_version(eco, str(ev["fixed"]).strip())
                flush()
            elif "last_affected" in ev and ev["last_affected"] is not None:
                cur_last = er.parse_version(eco, str(ev["last_affected"]).strip())
                flush()
        if has_open:
            flush()

        for intro_zero, intro_pv, fixed_pv, last_aff_pv in segments:
            if not intro_zero and intro_pv is None:
                continue
            for i, pv in enumerate(pvs):
                if pv.is_prerelease:
                    continue
                ge_intro = True if intro_zero else (pv.key >= intro_pv.key)
                if not ge_intro:
                    continue
                if fixed_pv is not None:
                    if pv.key < fixed_pv.key:
                        mask[i] = True
                elif last_aff_pv is not None:
                    if pv.key <= last_aff_pv.key:
                        mask[i] = True
                else:
                    # Unbounded introduced segment with no closing fixed/last_affected event
                    mask[i] = True
    return mask


def stratify_sample_advisories(rows: list[dict], cap: int, seed: int) -> list[dict]:
    """Calendar-year stratified sampling with exact inverse-probability weights N_h / n_h (cap <= 0 => 100% census)."""
    if cap <= 0 or len(rows) <= cap:
        for r in rows:
            r["weight"] = 1.0
        return rows
    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        yr = datetime.fromtimestamp(r["t_fix_sec"], tz=timezone.utc).year
        by_year[yr].append(r)
    years = sorted(by_year.keys())
    # Proportional allocation with floor >= 1 per non-empty year, summing cleanly to cap
    total = len(rows)
    alloc = {yr: max(1, int(round(cap * len(by_year[yr]) / total))) for yr in years}
    for yr in years:
        alloc[yr] = min(alloc[yr], len(by_year[yr]))
    # Adjust sum to exact cap
    while sum(alloc.values()) > cap:
        yr_max = max(years, key=lambda y: alloc[y])
        if alloc[yr_max] > 1:
            alloc[yr_max] -= 1
        else:
            break
    while sum(alloc.values()) < cap:
        cands = [y for y in years if alloc[y] < len(by_year[y])]
        if not cands:
            break
        yr_add = max(cands, key=lambda y: len(by_year[y]) - alloc[y])
        alloc[yr_add] += 1

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for yr in years:
        grp = sorted(by_year[yr], key=lambda x: (x["package_norm"], x["ghsa_id"]))
        n_h = alloc[yr]
        N_h = len(grp)
        if n_h >= N_h:
            chosen = grp
        else:
            idxs = sorted(rng.choice(N_h, size=n_h, replace=False).tolist())
            chosen = [grp[i] for i in idxs]
        w_h = N_h / float(len(chosen))
        for r in chosen:
            r["weight"] = w_h
            selected.append(r)
    return selected


def run_ecosystem(
    eco: str,
    adv_all_df: pl.DataFrame,
    mal_all_df: pl.DataFrame,
    cap: int,
    run_depth2: bool,
) -> dict:
    t0 = time.perf_counter()
    print(f"\n================ [{eco.upper()}] START (cap={cap}) ================", flush=True)
    p_files, v_files, d_files = get_parquet_paths(eco)
    audit: dict = {
        "ecosystem": eco,
        "pkg_parts": len(p_files),
        "ver_parts": len(v_files),
        "dep_parts": len(d_files),
    }

    # 1. Load packages
    t = time.perf_counter()
    pkgs_df = pl.read_parquet([str(p) for p in p_files], columns=["id", "name", "status"])
    audit["packages_total"] = pkgs_df.height
    norm_to_pids: dict[str, list[int]] = defaultdict(list)
    pid_to_norm: dict[int, str] = {}
    for pid, name in zip(pkgs_df["id"].to_list(), pkgs_df["name"].to_list()):
        if not name:
            continue
        pn = er.normalize_package_name(eco, name)
        norm_to_pids[pn].append(int(pid))
        pid_to_norm[int(pid)] = pn
    del pkgs_df
    print(f"  [{eco}] Loaded {audit['packages_total']:,} packages ({len(norm_to_pids):,} unique norm) in {time.perf_counter()-t:.1f}s", flush=True)

    # 2. Match advisories to upstream package_ids
    eco_adv_df = adv_all_df.filter(pl.col("ecosystem") == eco)
    audit["ghsa_fixed_raw"] = eco_adv_df.height
    adv_norms = set(eco_adv_df["package_norm"].to_list())
    matched_adv_pids: set[int] = set()
    for pn in adv_norms:
        matched_adv_pids.update(norm_to_pids.get(pn, []))

    # Also match MAL-* package_ids in the same versions scan!
    eco_mal_df = mal_all_df.filter(pl.col("ecosystem") == eco)
    audit["mal_records_raw"] = eco_mal_df.height
    mal_norms = set(eco_mal_df["package_norm"].to_list())
    matched_mal_pids: set[int] = set()
    for pn in mal_norms:
        matched_mal_pids.update(norm_to_pids.get(pn, []))

    target_ver_pids = matched_adv_pids | matched_mal_pids

    # 3. Fast Polars scan over versions: extract target upstream/MAL version strings AND compact
    #    (id, package_id, t_sec) table for vectorized t_next & active-package computation!
    t = time.perf_counter()
    target_pid_df = pl.DataFrame({"package_id": pl.Series(sorted(target_ver_pids), dtype=pl.Int64)})
    upstream_raw_vers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    total_vers_rows = 0
    null_ts_rows = 0
    ver_meta_chunks: list[pl.DataFrame] = []

    for i in range(0, len(v_files), 200):
        batch = [str(p) for p in v_files[i : i + 200]]
        vdf = pl.read_parquet(batch, columns=["id", "package_id", "number", "published_at_us"])
        total_vers_rows += vdf.height
        null_ts_rows += int((vdf["published_at_us"].is_null() | (vdf["published_at_us"] <= 0)).sum())

        # Extract target upstream & MAL-* version strings via fast semi-join
        if target_pid_df.height > 0:
            sub = vdf.join(target_pid_df, on="package_id", how="semi").filter(
                pl.col("number").is_not_null() & pl.col("published_at_us").is_not_null() & (pl.col("published_at_us") > 0)
            )
            if sub.height > 0:
                sub = sub.with_columns((pl.col("published_at_us") // 1_000_000).alias("t_sec")).filter(
                    (pl.col("t_sec") >= MIN_VER_SEC) & (pl.col("t_sec") <= CUTOFF_SEC)
                )
                for pid, num, tsec in zip(
                    sub["package_id"].to_list(), sub["number"].to_list(), sub["t_sec"].to_list()
                ):
                    upstream_raw_vers[pid_to_norm[int(pid)]].append((str(num), int(tsec)))

        # Keep compact valid (id, package_id, t_sec) chunk in RAM (only 24 bytes/row)
        v_valid = (
            vdf.select(["id", "package_id", "published_at_us"])
            .filter(pl.col("published_at_us").is_not_null() & (pl.col("published_at_us") > 0))
            .with_columns((pl.col("published_at_us") // 1_000_000).alias("t_sec"))
            .filter((pl.col("t_sec") >= MIN_VER_SEC) & (pl.col("t_sec") <= CUTOFF_SEC))
            .select(["id", "package_id", "t_sec"])
        )
        if v_valid.height > 0:
            ver_meta_chunks.append(v_valid)

    ver_all_df = pl.concat(ver_meta_chunks, rechunk=False) if ver_meta_chunks else pl.DataFrame(
        schema={"id": pl.Int64, "package_id": pl.Int64, "t_sec": pl.Int64}
    )
    del ver_meta_chunks
    audit["versions_total"] = total_vers_rows
    audit["versions_null_ts"] = null_ts_rows
    print(f"  [{eco}] Scanned {total_vers_rows:,} versions (valid={ver_all_df.height:,}, null_ts={null_ts_rows:,}) in {time.perf_counter()-t:.1f}s", flush=True)

    # 4. Evaluate empirical OpenSSF/OSV MAL-* detection-window curves (all, live 2024+, live 2025+, prev-legit)!
    mal_dwell_all: list[int] = []
    mal_dwell_live2024: list[int] = []
    mal_dwell_live2025: list[int] = []
    mal_dwell_prevlegit: list[int] = []
    mal_dwell_prevlegit_2024: list[int] = []
    mal_stats = Counter()
    for row in eco_mal_df.iter_rows(named=True):
        pn = row["package_norm"]
        t_osv = int(row["t_osv_pub_sec"])
        vlist = upstream_raw_vers.get(pn)
        if not vlist:
            mal_stats["unmatched_in_eco"] += 1
            continue
        ver_map = {}
        min_pkg_pub = min(ts for _, ts in vlist)
        for vnum, ts in vlist:
            if vnum not in ver_map or ts < ver_map[vnum]:
                ver_map[vnum] = ts
        aff_vers = json.loads(row["versions_json"])
        matched_ts = [ver_map[v] for v in aff_vers if v in ver_map]
        if not matched_ts and row["whole_package"]:
            matched_ts = [min_pkg_pub]
        if not matched_ts:
            mal_stats["ver_unmatched"] += 1
            continue
        t_mal_pub = min(matched_ts)
        dwell = t_osv - t_mal_pub
        if dwell < 0:
            mal_stats["negative_dwell_excluded"] += 1
            continue
        mal_stats["matched_valid"] += 1
        mal_dwell_all.append(dwell)
        is_live24 = t_mal_pub >= MAL_LIVE_2024_SEC
        is_live25 = t_mal_pub >= MAL_LIVE_2025_SEC
        is_prev_legit = (t_mal_pub - min_pkg_pub) >= 30 * eng.DAY
        if is_live24:
            mal_stats["matched_live_2024"] += 1
            mal_dwell_live2024.append(dwell)
        if is_live25:
            mal_stats["matched_live_2025"] += 1
            mal_dwell_live2025.append(dwell)
        if is_prev_legit:
            mal_stats["matched_prev_legit"] += 1
            mal_dwell_prevlegit.append(dwell)
            if is_live24:
                mal_stats["matched_prev_legit_2024"] += 1
                mal_dwell_prevlegit_2024.append(dwell)

    def _mal_curve(arr_list: list[int]) -> dict:
        arr = np.array(arr_list, dtype=np.int64)
        if len(arr) == 0:
            return {"n": 0, "median_days": None, "p25_days": None, "p75_days": None, "p90_days": None,
                    "blocking_pct": {str(k): None for k in eng.GATES}}
        return {
            "n": len(arr),
            "median_days": round(float(np.median(arr)) / eng.DAY, 3),
            "p25_days": round(float(np.percentile(arr, 25)) / eng.DAY, 3),
            "p75_days": round(float(np.percentile(arr, 75)) / eng.DAY, 3),
            "p90_days": round(float(np.percentile(arr, 90)) / eng.DAY, 3),
            "blocking_pct": {
                str(k): (0.0 if k == 0 else round(float(np.mean(arr <= k * eng.DAY) * 100.0), 3))
                for k in eng.GATES
            },
        }

    c_all = _mal_curve(mal_dwell_all)
    c_l24 = _mal_curve(mal_dwell_live2024)
    c_l25 = _mal_curve(mal_dwell_live2025)
    c_pl = _mal_curve(mal_dwell_prevlegit)
    c_pl24 = _mal_curve(mal_dwell_prevlegit_2024)
    mal_summary = {
        "counts": dict(mal_stats),
        "n_valid_all": c_all["n"],
        "n_valid_live_2024": c_l24["n"],
        "n_valid_live_2025": c_l25["n"],
        "n_valid_prev_legit": c_pl["n"],
        "n_valid_prev_legit_2024": c_pl24["n"],
        "median_days_all": c_all["median_days"],
        "p25_days_all": c_all["p25_days"],
        "p75_days_all": c_all["p75_days"],
        "p90_days_all": c_all["p90_days"],
        "median_days_live_2024": c_l24["median_days"],
        "p25_days_live_2024": c_l24["p25_days"],
        "p75_days_live_2024": c_l24["p75_days"],
        "p90_days_live_2024": c_l24["p90_days"],
        "median_days_live_2025": c_l25["median_days"],
        "median_days_prev_legit": c_pl["median_days"],
        "median_days_prev_legit_2024": c_pl24["median_days"],
        "blocking_all_pct": c_all["blocking_pct"],
        "blocking_live_2024_pct": c_l24["blocking_pct"],
        "blocking_live_2025_pct": c_l25["blocking_pct"],
        "blocking_prev_legit_pct": c_pl["blocking_pct"],
        "blocking_prev_legit_2024_pct": c_pl24["blocking_pct"],
    }
    print(
        f"  [{eco}] MAL-* valid={c_all['n']:,} (live24={c_l24['n']:,} med={c_l24['median_days']}d M7={c_l24['blocking_pct']['7']}%, "
        f"live25={c_l25['n']:,} med={c_l25['median_days']}d M7={c_l25['blocking_pct']['7']}%, "
        f"prev_legit={c_pl['n']:,} med={c_pl['median_days']}d M7={c_pl['blocking_pct']['7']}%)",
        flush=True,
    )

    # 5. Build parsed UpstreamData for every upstream with fixed advisories & filter eligible advisories
    upstream_objs: dict[str, tuple[eng.UpstreamData, list[er.ParsedVersion], dict[str, int]]] = {}
    for pn in adv_norms:
        vlist = upstream_raw_vers.get(pn)
        if not vlist:
            continue
        # Deduplicate version strings keeping earliest valid publish time
        vmap: dict[str, int] = {}
        for vnum, ts in vlist:
            if vnum not in vmap or ts < vmap[vnum]:
                vmap[vnum] = ts
        parsed_items: list[tuple[er.ParsedVersion, int]] = []
        for vnum, ts in vmap.items():
            pv = er.parse_version(eco, vnum)
            if pv is not None:
                parsed_items.append((pv, ts))
        if not parsed_items:
            continue
        parsed_items.sort(key=lambda x: (x[0].key, x[1], x[0].raw))
        pvs = [x[0] for x in parsed_items]
        t_pub = np.array([x[1] for x in parsed_items], dtype=np.int64)
        rank = np.arange(len(pvs), dtype=np.int64)
        is_pre = np.array([pv.is_prerelease for pv in pvs], dtype=bool)
        raw_to_idx = {pv.raw: idx for idx, pv in enumerate(pvs)}
        # Also map normalized key -> idx for exact matching (e.g. 'v1.2.3' vs '1.2.3', '1.2' vs '1.2.0')
        # Keep earliest publication index when canonical duplicates exist
        key_to_idx: dict[tuple, int] = {}
        for idx, pv in enumerate(pvs):
            if pv.key not in key_to_idx:
                key_to_idx[pv.key] = idx
        uobj = eng.UpstreamData(eco, pn, [pv.raw for pv in pvs], t_pub, rank, is_pre)
        upstream_objs[pn] = (uobj, pvs, {"raw": raw_to_idx, "key": key_to_idx})

    eligible_advs: list[dict] = []
    link_drop = Counter()
    for row in eco_adv_df.iter_rows(named=True):
        pn = row["package_norm"]
        if pn not in upstream_objs:
            link_drop["pkg_not_in_eco"] += 1
            continue
        uobj, pvs, idx_maps = upstream_objs[pn]
        fixed_raws = json.loads(row["fixed_versions_json"])
        fixed_indices = []
        for fx in fixed_raws:
            cand_vi = None
            if fx in idx_maps["raw"]:
                cand_vi = idx_maps["raw"][fx]
            else:
                fpv = er.parse_version(eco, fx)
                if fpv is not None and fpv.key in idx_maps["key"]:
                    cand_vi = idx_maps["key"][fpv.key]
            # Exclude prerelease fixed versions, matching standard non-prerelease resolver policy
            if cand_vi is not None and not uobj.is_prerelease[cand_vi]:
                fixed_indices.append(cand_vi)
        fixed_indices = sorted(set(fixed_indices))
        if not fixed_indices:
            link_drop["fixed_ver_not_found"] += 1
            continue
        t_fix_sec = int(min(uobj.t_pub[vi] for vi in fixed_indices))
        if t_fix_sec < FIX_START_SEC or t_fix_sec >= CUTOFF_SEC:
            link_drop["fix_outside_2018_2026"] += 1
            continue
        ranges_list = json.loads(row["ranges_json"])
        vmask = vuln_mask_for_ranges(eco, pvs, ranges_list)
        if not vmask.any():
            link_drop["empty_vuln_set"] += 1
            continue
        # Ensure fixed version itself is not marked vulnerable
        for vi in fixed_indices:
            vmask[vi] = False
        if not vmask.any():
            link_drop["empty_vuln_after_fix_excl"] += 1
            continue
        eligible_advs.append(
            {
                "ecosystem": eco,
                "ghsa_id": row["ghsa_id"],
                "native_id": row["native_id"],
                "package_norm": pn,
                "package_raw": row["package_raw"],
                "severity": row["severity"],
                "t_fix_sec": t_fix_sec,
                "t_ghsa_sec": int(row["t_ghsa_sec"]),
                "t_osv_earliest_sec": int(row["t_osv_earliest_sec"]),
                "vendor_lag_sec": int(row["vendor_lag_sec"]),
                "fixed_indices": fixed_indices,
                "fixed_versions": [uobj.versions[vi] for vi in fixed_indices],
                "vuln_mask": vmask,
            }
        )

    audit["eligible_linked_advisories"] = len(eligible_advs)
    audit["link_drops"] = dict(link_drop)
    selected_advs = stratify_sample_advisories(eligible_advs, cap=cap, seed=SEED)
    audit["selected_advisories"] = len(selected_advs)
    selected_upstreams = sorted({a["package_norm"] for a in selected_advs})
    audit["selected_unique_upstreams"] = len(selected_upstreams)
    print(
        f"  [{eco}] Eligible advisories={len(eligible_advs):,}, selected={len(selected_advs):,} "
        f"({len(selected_upstreams):,} upstreams), drops={dict(link_drop)}",
        flush=True,
    )

    # 6. Vectorized Polars scan over dependencies/<eco>/*.parquet targeting `selected_upstreams`!
    t = time.perf_counter()
    target_up_df = pl.DataFrame({"_pn": pl.Series(selected_upstreams, dtype=pl.String)})
    dep_chunks: list[pl.DataFrame] = []
    dep_total_rows = 0
    for i in range(0, len(d_files), 200):
        batch = [str(p) for p in d_files[i : i + 200]]
        ddf = pl.read_parquet(batch, columns=["version_id", "package_name", "requirements", "optional"])
        dep_total_rows += ddf.height
        if eco == "pypi":
            ddf = ddf.with_columns(
                pl.col("package_name").str.strip_chars().str.to_lowercase().str.replace_all(r"[-_.]+", "-").alias("_pn")
            )
        elif eco == "cargo":
            ddf = ddf.with_columns(
                pl.col("package_name").str.strip_chars().str.to_lowercase().str.replace_all("_", "-").alias("_pn")
            )
        elif eco == "packagist":
            ddf = ddf.with_columns(
                pl.col("package_name").str.strip_chars().str.to_lowercase().alias("_pn")
            )
        else:
            ddf = ddf.with_columns(pl.col("package_name").str.strip_chars().alias("_pn"))
        sub = ddf.join(target_up_df, on="_pn", how="semi").filter(
            pl.col("version_id").is_not_null() & (pl.col("version_id") > 0)
        ).select([
            pl.col("version_id"),
            pl.col("_pn"),
            pl.col("requirements").fill_null("").cast(pl.String),
            pl.col("optional").fill_null(False).cast(pl.Boolean),
        ])
        if sub.height > 0:
            dep_chunks.append(sub)

    if dep_chunks:
        dep_all_df = pl.concat(dep_chunks, rechunk=False)
        del dep_chunks
        # Optional overrides runtime when duplicate (version_id, _pn) exists
        dep_all_df = (
            dep_all_df.sort("optional", descending=True)
            .unique(subset=["version_id", "_pn"], keep="first")
            .select(["version_id", "_pn", "requirements"])
        )
    else:
        dep_all_df = pl.DataFrame(schema={"version_id": pl.Int64, "_pn": pl.String, "requirements": pl.String})

    audit["dependencies_total_rows"] = dep_total_rows
    audit["matched_raw_dep_edges"] = dep_all_df.height
    print(f"  [{eco}] Scanned {dep_total_rows:,} dep rows -> {dep_all_df.height:,} matched edges in {time.perf_counter()-t:.1f}s", flush=True)

    # 7. Vectorized Polars resolution of dependent version_id -> (dep_pkg_id, t_d, t_next, is_active)!
    t = time.perf_counter()
    needed_vids_df = dep_all_df.select(pl.col("version_id").alias("id")).unique()
    matched_vers = ver_all_df.join(needed_vids_df, on="id", how="semi")
    needed_pids_df = matched_vers.select("package_id").unique()

    # Collect all valid releases of needed dependent packages to compute exact is_active & t_next!
    dep_pkg_vers = ver_all_df.join(needed_pids_df, on="package_id", how="semi")
    vid_to_pid_df = ver_all_df.select([pl.col("id").alias("version_id"), "package_id"]) if run_depth2 else None
    del ver_all_df

    pkg_act_df = dep_pkg_vers.group_by("package_id").len().with_columns((pl.col("len") >= 2).alias("is_active")).select(["package_id", "is_active"])
    uniq_pkg_ts = (
        dep_pkg_vers.select(["package_id", "t_sec"])
        .unique()
        .sort(["package_id", "t_sec"])
        .with_columns(
            pl.col("t_sec").shift(-1).over("package_id").fill_null(CUTOFF_SEC).alias("t_next")
        )
    )
    del dep_pkg_vers

    vid_meta = (
        matched_vers.join(uniq_pkg_ts, on=["package_id", "t_sec"], how="left")
        .join(pkg_act_df, on="package_id", how="left")
        .with_columns(
            pl.when(pl.col("t_next").is_null() | (pl.col("t_next") <= pl.col("t_sec")))
            .then(CUTOFF_SEC)
            .otherwise(pl.col("t_next"))
            .alias("t_next"),
            pl.col("is_active").fill_null(False),
        )
        .select(["id", "package_id", "t_sec", "t_next", "is_active"])
    )
    del matched_vers, uniq_pkg_ts, pkg_act_df

    live_df = dep_all_df.join(vid_meta, left_on="version_id", right_on="id", how="inner")
    dropped_no_ts = int(dep_all_df.height - live_df.height)
    del dep_all_df, vid_meta

    # Partition live_df by upstream into compact NumPy arrays (zero millions-of-tuples Python overhead!)
    rels_by_up: dict[str, dict] = {}
    for (up_norm,), grp in live_df.partition_by("_pn", as_dict=True).items():
        rels_by_up[up_norm] = {
            "dep_pid": grp["package_id"].to_numpy(),
            "td": grp["t_sec"].to_numpy(),
            "t_next": grp["t_next"].to_numpy(),
            "reqs": grp["requirements"].to_list(),
            "is_act": grp["is_active"].to_numpy(),
        }
    audit["live_relations_total"] = int(live_df.height)
    audit["live_relations_dropped_bad_ts"] = dropped_no_ts
    del live_df
    print(f"  [{eco}] Built {audit['live_relations_total']:,} live relations (dropped_ts={dropped_no_ts:,}) in {time.perf_counter()-t:.1f}s", flush=True)

    # 8. Core Vectorized Evaluation across all selected advisories!
    t = time.perf_counter()
    advs_by_up: dict[str, list[dict]] = defaultdict(list)
    for a in selected_advs:
        advs_by_up[a["package_norm"]].append(a)

    # Per-advisory output rows (for bootstrap, Shapley, top-3 hub deletion, recent-era, median/IQR/p90!)
    adv_records: list[dict] = []
    # Openness counters
    openness_counts = Counter()
    unparseable_rels = 0
    total_eval_rels = 0

    # Also accumulate fresh & sync unavailability across ALL unique relations per upstream (deduped across advisories)
    unavail_acc = {
        stratum: {
            k: {
                "n_resolved_p0_fresh": 0,
                "n_unavail_fresh": 0,
                "fresh_unavail_sec": 0,
                "fresh_live_sec": 0,
                "n_resolved_p0_sync": 0,
                "n_unavail_sync": 0,
                "sync_unavail_sec": 0,
                "sync_live_sec": 0,
            }
            for k in eng.GATES
        }
        for stratum in ("full", "active")
    }
    # Per-upstream unavailability for clustered bootstrap
    up_unavail_rows: list[dict] = []

    for up_norm, adv_list in advs_by_up.items():
        rels = rels_by_up.get(up_norm)
        if not rels:
            continue
        uobj, pvs, _ = upstream_objs[up_norm]
        V = len(pvs)
        reqs_list = rels["reqs"]
        n_raw_rels = len(reqs_list)

        # Compile distinct constraints for this upstream ONCE
        uniq_reqs = sorted(set(reqs_list))
        req_to_cidx: dict[str, int] = {}
        c_admitted_list: list[list[bool]] = []
        c_openness_list: list[str] = []
        for rq in uniq_reqs:
            fn, fam = er.compile_constraint(eco, rq)
            if fn is None:
                req_to_cidx[rq] = -1
            else:
                req_to_cidx[rq] = len(c_admitted_list)
                c_admitted_list.append(fn(pvs))
                c_openness_list.append(fam)
        if not c_admitted_list:
            unparseable_rels += n_raw_rels
            continue

        admitted = np.array(c_admitted_list, dtype=bool)
        c_openness = np.array(c_openness_list)

        # Vectorized filter over compact NumPy arrays!
        raw_c = np.array([req_to_cidx[rq] for rq in reqs_list], dtype=np.int64)
        valid_mask = raw_c >= 0
        unparseable_rels += int(n_raw_rels - valid_mask.sum())
        if not valid_mask.any():
            continue
        r_c = raw_c[valid_mask]
        r_td = rels["td"][valid_mask]
        r_tnext = rels["t_next"][valid_mask]
        r_act = rels["is_act"][valid_mask]
        r_fam = c_openness[r_c]
        total_eval_rels += len(r_c)
        # Fast vectorized openness counts
        bc = np.bincount(r_c, minlength=len(c_openness_list))
        for ci_idx, cnt_i in enumerate(bc):
            if cnt_i > 0:
                openness_counts[c_openness_list[ci_idx]] += int(cnt_i)

        # Precompute P_k no-exemption step functions ONCE per upstream (for k in GATES)!
        steps_noex = {}
        for k in eng.GATES:
            ek = eng.compute_eligibility(uobj.t_pub, k_days=k)
            steps_noex[k] = eng.build_step_function(uobj.rank, admitted, ek)

        # Compute fresh & synchronized unavailability once per upstream (no double counting across advisories!)
        T0, V0 = steps_noex[0]
        val0_at_td = eng.value_at_time(T0, V0, r_c, r_td)
        res0_fresh = val0_at_td >= 0
        dummy_ind = np.zeros(V, dtype=bool)

        up_u_rec = {"upstream": up_norm, "n_rels": int(len(r_c))}
        for k in eng.GATES:
            Tk, Vk = steps_noex[k]
            valk_at_td = eng.value_at_time(Tk, Vk, r_c, r_td)
            u_fresh_hit = res0_fresh & (valk_at_td < 0)
            # Negative control assertion at k=0!
            if k == 0 and u_fresh_hit.any():
                raise RuntimeError(f"Negative control k=0 failed fresh unavailability on {eco}:{up_norm}")
            u_fresh_sec = eng.integrate_indicator(Tk, Vk, dummy_ind, r_c, r_td, r_tnext, include_unavail=True)
            u0_fresh_sec = eng.integrate_indicator(T0, V0, dummy_ind, r_c, r_td, r_tnext, include_unavail=True)
            added_fresh_u_sec = np.maximum(0, u_fresh_sec - u0_fresh_sec)
            live_fresh_sec = np.maximum(0, r_tnext - r_td)

            # Synchronized unavailability on [t_d + k, t_next)
            td_sync = r_td + np.int64(k) * eng.DAY
            sync_valid = r_tnext > td_sync
            val0_sync = eng.value_at_time(T0, V0, r_c[sync_valid], td_sync[sync_valid]) if sync_valid.any() else np.zeros(0, dtype=np.int64)
            valk_sync = eng.value_at_time(Tk, Vk, r_c[sync_valid], td_sync[sync_valid]) if sync_valid.any() else np.zeros(0, dtype=np.int64)
            res0_sync_sub = val0_sync >= 0
            u_sync_hit_sub = res0_sync_sub & (valk_sync < 0)
            if sync_valid.any():
                u_sync_sec_sub = eng.integrate_indicator(
                    Tk, Vk, dummy_ind, r_c[sync_valid], td_sync[sync_valid], r_tnext[sync_valid], include_unavail=True
                )
                u0_sync_sec_sub = eng.integrate_indicator(
                    T0, V0, dummy_ind, r_c[sync_valid], td_sync[sync_valid], r_tnext[sync_valid], include_unavail=True
                )
                added_sync_u_sec_sub = np.maximum(0, u_sync_sec_sub - u0_sync_sec_sub)
                live_sync_sec_sub = r_tnext[sync_valid] - td_sync[sync_valid]
                act_sync_sub = r_act[sync_valid]
            else:
                added_sync_u_sec_sub = np.zeros(0, dtype=np.int64)
                live_sync_sec_sub = np.zeros(0, dtype=np.int64)
                act_sync_sub = np.zeros(0, dtype=bool)

            for stratum, smask, smask_sync in (
                ("full", np.ones(len(r_c), dtype=bool), np.ones(len(act_sync_sub), dtype=bool)),
                ("active", r_act, act_sync_sub),
            ):
                acc = unavail_acc[stratum][k]
                acc["n_resolved_p0_fresh"] += int((res0_fresh & smask).sum())
                acc["n_unavail_fresh"] += int((u_fresh_hit & smask).sum())
                acc["fresh_unavail_sec"] += int(added_fresh_u_sec[smask].sum())
                acc["fresh_live_sec"] += int(live_fresh_sec[smask].sum())
                acc["n_resolved_p0_sync"] += int((res0_sync_sub & smask_sync).sum())
                acc["n_unavail_sync"] += int((u_sync_hit_sub & smask_sync).sum())
                acc["sync_unavail_sec"] += int(added_sync_u_sec_sub[smask_sync].sum())
                acc["sync_live_sec"] += int(live_sync_sec_sub[smask_sync].sum())

            if k == 7:
                up_u_rec["fresh_res0_7"] = int(res0_fresh.sum())
                up_u_rec["fresh_hit_7"] = int(u_fresh_hit.sum())
                up_u_rec["sync_res0_7"] = int(res0_sync_sub.sum())
                up_u_rec["sync_hit_7"] = int(u_sync_hit_sub.sum())
                up_u_rec["fresh_res0_7_act"] = int((res0_fresh & r_act).sum())
                up_u_rec["fresh_hit_7_act"] = int((u_fresh_hit & r_act).sum())
        up_unavail_rows.append(up_u_rec)

        # Evaluate each advisory on this upstream!
        for adv in adv_list:
            t_f = int(adv["t_fix_sec"])
            x = np.maximum(r_td, t_f)
            y = np.minimum(r_tnext, CUTOFF_SEC)
            live = y > x
            if not live.any():
                continue
            lc = r_c[live]
            lx = x[live]
            ly = y[live]
            lact = r_act[live]
            lfam = r_fam[live]
            vmask = adv["vuln_mask"]
            fidxs = adv["fixed_indices"]
            w_f = float(adv["weight"])

            # Check how many live relations admit at least one fixed version under P0
            admits_fix = admitted[lc][:, fidxs].any(axis=1)
            n_live = int(len(lc))
            n_admit_fix = int(admits_fix.sum())

            # Baseline R(P0)
            r0_sec = eng.integrate_indicator(T0, V0, vmask, lc, lx, ly, include_unavail=False)
            r0_days_full = float(r0_sec.sum()) / eng.DAY
            r0_days_act = float(r0_sec[lact].sum()) / eng.DAY

            rec = {
                "ecosystem": eco,
                "ghsa_id": adv["ghsa_id"],
                "native_id": adv["native_id"],
                "package_norm": up_norm,
                "severity": adv["severity"],
                "t_fix_sec": t_f,
                "fix_year": datetime.fromtimestamp(t_f, tz=timezone.utc).year,
                "t_ghsa_sec": adv["t_ghsa_sec"],
                "t_osv_earliest_sec": adv["t_osv_earliest_sec"],
                "lag_ghsa_sec": adv["t_ghsa_sec"] - t_f,
                "lag_osv_sec": adv["t_osv_earliest_sec"] - t_f,
                "vendor_lag_sec": adv["vendor_lag_sec"],
                "weight": w_f,
                "n_live_rels": n_live,
                "n_live_rels_act": int(lact.sum()),
                "n_admit_fix": n_admit_fix,
                "r0_days_full": r0_days_full,
                "r0_days_act": r0_days_act,
            }

            for k in eng.GATES:
                Tk, Vk = steps_noex[k]
                rk_no_sec = eng.integrate_indicator(Tk, Vk, vmask, lc, lx, ly, include_unavail=False)
                add_no_sec = rk_no_sec - r0_sec
                if k == 0 and int(np.abs(add_no_sec).sum()) != 0:
                    raise RuntimeError(f"Negative control k=0 failed exposure on {eco}:{adv['ghsa_id']}")

                if k == 0:
                    add_ghsa_sec = add_no_sec
                    add_osv_sec = add_no_sec
                    add_zero_sec = add_no_sec
                else:
                    e_ghsa = eng.compute_eligibility(uobj.t_pub, k, fidxs, advisory_sec=adv["t_ghsa_sec"])
                    Tg, Vg = eng.build_step_function(uobj.rank, admitted, e_ghsa)
                    add_ghsa_sec = eng.integrate_indicator(Tg, Vg, vmask, lc, lx, ly, include_unavail=False) - r0_sec

                    if adv["t_osv_earliest_sec"] == adv["t_ghsa_sec"]:
                        add_osv_sec = add_ghsa_sec
                    else:
                        e_osv = eng.compute_eligibility(uobj.t_pub, k, fidxs, advisory_sec=adv["t_osv_earliest_sec"])
                        To, Vo = eng.build_step_function(uobj.rank, admitted, e_osv)
                        add_osv_sec = eng.integrate_indicator(To, Vo, vmask, lc, lx, ly, include_unavail=False) - r0_sec

                    # Conservative +12h end-of-day sensitivity for date-only timestamps (exact 12:00:00Z, e.g. RUSTSEC)
                    if k == 7 and adv["t_osv_earliest_sec"] != adv["t_ghsa_sec"] and (adv["t_osv_earliest_sec"] % eng.DAY) == 43200:
                        t_osv_eod = min(adv["t_ghsa_sec"], adv["t_osv_earliest_sec"] + 43199)
                        e_osv_eod = eng.compute_eligibility(uobj.t_pub, k, fidxs, advisory_sec=t_osv_eod)
                        Toe, Voe = eng.build_step_function(uobj.rank, admitted, e_osv_eod)
                        add_osv_eod_sec = eng.integrate_indicator(Toe, Voe, vmask, lc, lx, ly, include_unavail=False) - r0_sec
                    else:
                        add_osv_eod_sec = add_osv_sec

                    e_zero = eng.compute_eligibility(uobj.t_pub, k, fidxs, ideal_zero_lag=True)
                    Tz, Vz = eng.build_step_function(uobj.rank, admitted, e_zero)
                    add_zero_sec = eng.integrate_indicator(Tz, Vz, vmask, lc, lx, ly, include_unavail=False) - r0_sec

                rec[f"add_no_{k}_full"] = float(add_no_sec.sum()) / eng.DAY
                rec[f"add_ghsa_{k}_full"] = float(add_ghsa_sec.sum()) / eng.DAY
                rec[f"add_osv_{k}_full"] = float(add_osv_sec.sum()) / eng.DAY
                if k == 7:
                    rec["add_osv_eod_7_full"] = float(add_osv_eod_sec.sum()) / eng.DAY
                rec[f"add_zero_{k}_full"] = float(add_zero_sec.sum()) / eng.DAY

                rec[f"add_no_{k}_act"] = float(add_no_sec[lact].sum()) / eng.DAY
                rec[f"add_ghsa_{k}_act"] = float(add_ghsa_sec[lact].sum()) / eng.DAY
                rec[f"add_osv_{k}_act"] = float(add_osv_sec[lact].sum()) / eng.DAY
                rec[f"add_zero_{k}_act"] = float(add_zero_sec[lact].sum()) / eng.DAY

                # Store 4-class openness cell totals at k=7 for exact Shapley decomposition!
                if k == 7:
                    for fam in eng.OPENNESS_CLASSES:
                        fm = lfam == fam
                        rec[f"c7_{fam}_no"] = float(add_no_sec[fm].sum()) / eng.DAY
                        rec[f"c7_{fam}_ghsa"] = float(add_ghsa_sec[fm].sum()) / eng.DAY
                        rec[f"c7_{fam}_osv"] = float(add_osv_sec[fm].sum()) / eng.DAY
                        rec[f"c7_{fam}_n"] = int(fm.sum())
            adv_records.append(rec)

    audit["advisories_with_live_rels"] = len(adv_records)
    audit["evaluated_unique_upstreams"] = len({r["package_norm"] for r in adv_records})
    audit["evaluated_relations_total"] = total_eval_rels
    audit["unparseable_relations"] = unparseable_rels
    audit["unparseable_pct"] = round(100.0 * unparseable_rels / max(total_eval_rels + unparseable_rels, 1), 4)
    audit["openness_counts"] = dict(openness_counts)
    print(
        f"  [{eco}] Evaluated {len(adv_records):,} advisories ({audit['evaluated_unique_upstreams']:,} upstreams) across {total_eval_rels:,} rels "
        f"(unparseable={audit['unparseable_pct']}%) in {time.perf_counter()-t:.1f}s",
        flush=True,
    )

    # 9. Reverse-Dependency Fan-Out Reweighting Check (50 positive-admission advisories per ecosystem)
    depth2_summary = None
    if run_depth2 and adv_records:
        t_d2 = time.perf_counter()
        rng_d2 = np.random.default_rng(SEED)
        # Pick up to 50 advisories with >= 1 live direct relation that admits the fix
        cands_d2 = [r for r in adv_records if r["n_admit_fix"] > 0]
        if len(cands_d2) > 50:
            idxs = sorted(rng_d2.choice(len(cands_d2), size=50, replace=False).tolist())
            sample_d2 = [cands_d2[i] for i in idxs]
        else:
            sample_d2 = cands_d2
        # Find direct dependents (level-1 intermediates) of these upstreams
        d2_ups = {r["package_norm"] for r in sample_d2}
        inter_pkg_names: set[str] = set()
        for upn in d2_ups:
            rel_d = rels_by_up.get(upn)
            if not rel_d:
                continue
            for dpid in np.unique(rel_d["dep_pid"]).tolist():
                inm = pid_to_norm.get(int(dpid))
                if inm and inm != upn:
                    inter_pkg_names.add(inm)
        # Count UNIQUE official-registry reverse-dependent packages per intermediate (mirror-filtered via vid_to_pid_df)
        inter_rev_pkgs: dict[str, set[int]] = defaultdict(set)
        if inter_pkg_names and vid_to_pid_df is not None:
            inter_pkg_df = pl.DataFrame({"_pn": pl.Series(sorted(inter_pkg_names), dtype=pl.String)})
            for i in range(0, len(d_files), 200):
                batch = [str(p) for p in d_files[i : i + 200]]
                ddf = pl.read_parquet(batch, columns=["version_id", "package_name"])
                if eco == "pypi":
                    ddf = ddf.with_columns(
                        pl.col("package_name").str.strip_chars().str.to_lowercase().str.replace_all(r"[-_.]+", "-").alias("_pn")
                    )
                elif eco == "cargo":
                    ddf = ddf.with_columns(
                        pl.col("package_name").str.strip_chars().str.to_lowercase().str.replace_all("_", "-").alias("_pn")
                    )
                elif eco == "packagist":
                    ddf = ddf.with_columns(
                        pl.col("package_name").str.strip_chars().str.to_lowercase().alias("_pn")
                    )
                else:
                    ddf = ddf.with_columns(pl.col("package_name").str.strip_chars().alias("_pn"))
                sub = (
                    ddf.join(inter_pkg_df, on="_pn", how="semi")
                    .select(["_pn", "version_id"])
                    .join(vid_to_pid_df, on="version_id", how="inner")
                    .select(["_pn", "package_id"])
                    .unique()
                )
                for inm, rpid in zip(sub["_pn"].to_list(), sub["package_id"].to_list()):
                    inter_rev_pkgs[inm].add(int(rpid))
        inter_dep_counts = {inm: len(s) for inm, s in inter_rev_pkgs.items()}
        del inter_rev_pkgs
        # Evaluate each unique intermediate package ONCE per advisory (using its latest live relation at/after t_fix)
        tot_d1_pkgs = 0
        tot_d2_fanout = 0
        tot_d1_admit = 0
        tot_d2_admit_weighted = 0
        for r in sample_d2:
            upn = r["package_norm"]
            rel_d = rels_by_up.get(upn)
            if not rel_d:
                continue
            adv_obj = next(a for a in advs_by_up[upn] if a["ghsa_id"] == r["ghsa_id"])
            uobj, pvs, _ = upstream_objs[upn]
            fidxs = adv_obj["fixed_indices"]
            rq_admits_cache: dict[str, bool | None] = {}
            dpids_arr = rel_d["dep_pid"]
            tds_arr = rel_d["td"]
            tns_arr = rel_d["t_next"]
            rqs_arr = rel_d["reqs"]
            live_idx = np.nonzero(np.minimum(tns_arr, CUTOFF_SEC) > np.maximum(tds_arr, adv_obj["t_fix_sec"]))[0]
            # Keep the latest live relation per unique intermediate dep_pid for this advisory
            latest_by_dpid: dict[int, tuple[int, str]] = {}
            for j in live_idx.tolist():
                dpid = int(dpids_arr[j])
                td_j = int(tds_arr[j])
                prev = latest_by_dpid.get(dpid)
                if prev is None or td_j > prev[0]:
                    latest_by_dpid[dpid] = (td_j, rqs_arr[j])
            for dpid, (_td, rq) in latest_by_dpid.items():
                if rq not in rq_admits_cache:
                    fn, _fam = er.compile_constraint(eco, rq)
                    if fn is None:
                        rq_admits_cache[rq] = None
                    else:
                        adm = fn(pvs)
                        rq_admits_cache[rq] = any(adm[fi] for fi in fidxs)
                admits = rq_admits_cache[rq]
                if admits is None:
                    continue
                inm = pid_to_norm.get(dpid, "")
                d2_cnt = inter_dep_counts.get(inm, 0)
                tot_d1_pkgs += 1
                tot_d2_fanout += d2_cnt
                if admits:
                    tot_d1_admit += 1
                    tot_d2_admit_weighted += d2_cnt
        depth2_summary = {
            "sample_advisories": len(sample_d2),
            "unique_intermediates": len(inter_pkg_names),
            "direct_unique_intermediates": tot_d1_pkgs,
            "fanout_weighted_pkgs": tot_d2_fanout,
            "d1_fix_admit_pct": round(100.0 * tot_d1_admit / max(tot_d1_pkgs, 1), 2),
            "d2_fanout_weighted_admit_pct": round(100.0 * tot_d2_admit_weighted / max(tot_d2_fanout, 1), 2),
            "mean_rev_dep_fanout": round(tot_d2_fanout / max(tot_d1_pkgs, 1), 2),
            "elapsed_sec": round(time.perf_counter() - t_d2, 1),
        }
        print(f"  [{eco}] Fan-out reweighting sample: {depth2_summary}", flush=True)

    # 10. Export 20 stratified advisories per ecosystem for the 100-record manual audit!
    rng_aud = np.random.default_rng(SEED + len(eco))
    manual_sample = []
    if adv_records:
        k_aud = min(20, len(adv_records))
        idxs = sorted(rng_aud.choice(len(adv_records), size=k_aud, replace=False).tolist())
        for i in idxs:
            r = adv_records[i]
            manual_sample.append(
                {
                    "ecosystem": eco,
                    "ghsa_id": r["ghsa_id"],
                    "native_id": r["native_id"],
                    "package_norm": r["package_norm"],
                    "severity": r["severity"],
                    "fix_utc": datetime.fromtimestamp(r["t_fix_sec"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ghsa_utc": datetime.fromtimestamp(r["t_ghsa_sec"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "osv_utc": datetime.fromtimestamp(r["t_osv_earliest_sec"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lag_ghsa_days": round(r["lag_ghsa_sec"] / eng.DAY, 2),
                    "lag_osv_days": round(r["lag_osv_sec"] / eng.DAY, 2),
                    "vendor_lag_days": round(r["vendor_lag_sec"] / eng.DAY, 2),
                    "n_live_rels": r["n_live_rels"],
                    "add_no_7_days": round(r["add_no_7_full"], 2),
                    "add_ghsa_7_days": round(r["add_ghsa_7_full"], 2),
                    "add_osv_7_days": round(r["add_osv_7_full"], 2),
                }
            )

    audit["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    return {
        "ecosystem": eco,
        "audit": audit,
        "mal_summary": mal_summary,
        "unavail_acc": unavail_acc,
        "up_unavail_rows": up_unavail_rows,
        "adv_records": adv_records,
        "depth2_summary": depth2_summary,
        "manual_sample": manual_sample,
    }


def summarize_and_bootstrap(eco_results: list[dict], mode: str, n_boot: int) -> dict:
    """Compute all headline tables, CIs, analytical unweighted baselines, Shapley decomposition, and sensitivity checks."""
    out_summary: dict = {
        "mode": mode,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_boot": n_boot,
        "seed": SEED,
        "ecosystems": {},
    }

    # First pass: compute per-ecosystem aggregates + openness cells for Shapley reference
    cell_data_by_eco = {}
    for er_res in eco_results:
        eco = er_res["ecosystem"]
        recs = er_res["adv_records"]
        u_acc = er_res["unavail_acc"]
        up_u = er_res["up_unavail_rows"]

        w = np.array([r["weight"] for r in recs], dtype=np.float64)
        cl_ids = np.array([r["package_norm"] for r in recs])
        lag_ghsa = np.array([r["lag_ghsa_sec"] for r in recs], dtype=np.float64)
        lag_osv = np.array([r["lag_osv_sec"] for r in recs], dtype=np.float64)
        v_lag = np.array([r["vendor_lag_sec"] for r in recs], dtype=np.float64)
        fix_yr = np.array([r["fix_year"] for r in recs], dtype=np.int64)
        recent_mask = fix_yr >= 2022

        # Top-3 upstream hubs by unweighted no-exemption 7-day added exposure
        up_totals = defaultdict(float)
        for r in recs:
            up_totals[r["package_norm"]] += r["weight"] * r["add_no_7_full"]
        top3_ups = [u for u, _ in sorted(up_totals.items(), key=lambda x: (-x[1], x[0]))[:3]]
        not_top3 = np.array([r["package_norm"] not in top3_ups for r in recs], dtype=bool)
        tot_w7 = sum(up_totals.values())
        top3_share_pct = round(100.0 * sum(up_totals[u] for u in top3_ups) / max(tot_w7, 1e-12), 2)

        gates_summary = {}
        for k in eng.GATES:
            a_no_f = w * np.array([r[f"add_no_{k}_full"] for r in recs])
            a_gh_f = w * np.array([r[f"add_ghsa_{k}_full"] for r in recs])
            a_os_f = w * np.array([r[f"add_osv_{k}_full"] for r in recs])
            a_z0_f = w * np.array([r[f"add_zero_{k}_full"] for r in recs])

            a_no_a = w * np.array([r[f"add_no_{k}_act"] for r in recs])
            a_gh_a = w * np.array([r[f"add_ghsa_{k}_act"] for r in recs])
            a_os_a = w * np.array([r[f"add_osv_{k}_act"] for r in recs])

            # Exemption recovery (%) with common random numbers (CRN) clustered bootstrap CIs!
            rec_gh_f, rec_gh_f_lo, rec_gh_f_hi = eng.clustered_bootstrap_ci(
                cl_ids, a_no_f - a_gh_f, a_no_f, n_boot=n_boot, seed=SEED + k, scale=100.0
            )
            rec_os_f, rec_os_f_lo, rec_os_f_hi = eng.clustered_bootstrap_ci(
                cl_ids, a_no_f - a_os_f, a_no_f, n_boot=n_boot, seed=SEED + k, scale=100.0
            )
            feed_gain_f, fg_lo, fg_hi = eng.clustered_bootstrap_ci(
                cl_ids, a_gh_f - a_os_f, a_no_f, n_boot=n_boot, seed=SEED + k, scale=100.0
            )
            rec_z0_f = (float((a_no_f - a_z0_f).sum()) / max(float(a_no_f.sum()), 1e-12) * 100.0) if k > 0 else 0.0

            rec_gh_a, rec_gh_a_lo, rec_gh_a_hi = eng.clustered_bootstrap_ci(
                cl_ids, a_no_a - a_gh_a, a_no_a, n_boot=n_boot, seed=SEED + 300 + k, scale=100.0
            )
            rec_os_a, rec_os_a_lo, rec_os_a_hi = eng.clustered_bootstrap_ci(
                cl_ids, a_no_a - a_os_a, a_no_a, n_boot=n_boot, seed=SEED + 300 + k, scale=100.0
            )
            feed_gain_a, fga_lo, fga_hi = eng.clustered_bootstrap_ci(
                cl_ids, a_gh_a - a_os_a, a_no_a, n_boot=n_boot, seed=SEED + 300 + k, scale=100.0
            )

            # Analytical unweighted plug-in baseline (W10-style) + paired clustered bootstrap gap CIs
            b_unw_gh = eng.unweighted_plugin_recovery(lag_ghsa, k)
            b_unw_os = eng.unweighted_plugin_recovery(lag_osv, k)
            _, gap_gh_lo, gap_gh_hi = eng.clustered_bootstrap_gap_vs_unw_ci(
                cl_ids, a_no_f - a_gh_f, a_no_f, lag_ghsa, k, n_boot=n_boot, seed=SEED + k
            )
            _, gap_os_lo, gap_os_hi = eng.clustered_bootstrap_gap_vs_unw_ci(
                cl_ids, a_no_f - a_os_f, a_no_f, lag_osv, k, n_boot=n_boot, seed=SEED + k
            )

            # Recent era (2022-2026) and Top-3 hub deletion sensitivity at k=7
            rec_gh_recent = (
                float((a_no_f[recent_mask] - a_gh_f[recent_mask]).sum())
                / max(float(a_no_f[recent_mask].sum()), 1e-12)
                * 100.0
                if recent_mask.any() and k > 0
                else 0.0
            )
            rec_os_recent = (
                float((a_no_f[recent_mask] - a_os_f[recent_mask]).sum())
                / max(float(a_no_f[recent_mask].sum()), 1e-12)
                * 100.0
                if recent_mask.any() and k > 0
                else 0.0
            )
            rec_gh_notop3 = (
                float((a_no_f[not_top3] - a_gh_f[not_top3]).sum())
                / max(float(a_no_f[not_top3].sum()), 1e-12)
                * 100.0
                if not_top3.any() and k > 0
                else 0.0
            )
            rec_os_notop3 = (
                float((a_no_f[not_top3] - a_os_f[not_top3]).sum())
                / max(float(a_no_f[not_top3].sum()), 1e-12)
                * 100.0
                if not_top3.any() and k > 0
                else 0.0
            )

            # Per-advisory absolute added exposure days (median, IQR, p90) among advisories with >=1 admitting rel
            per_adv_add_no = np.array([r[f"add_no_{k}_full"] / max(r["n_admit_fix"], 1) for r in recs if r["n_admit_fix"] > 0])
            per_adv_add_gh = np.array([r[f"add_ghsa_{k}_full"] / max(r["n_admit_fix"], 1) for r in recs if r["n_admit_fix"] > 0])

            # Unavailability metrics
            uf = u_acc["full"][k]
            ua = u_acc["active"][k]
            fresh_pct_f = round(100.0 * uf["n_unavail_fresh"] / max(uf["n_resolved_p0_fresh"], 1), 3)
            sync_pct_f = round(100.0 * uf["n_unavail_sync"] / max(uf["n_resolved_p0_sync"], 1), 3)
            fresh_day_pct_f = round(100.0 * uf["fresh_unavail_sec"] / max(uf["fresh_live_sec"], 1), 4)
            sync_day_pct_f = round(100.0 * uf["sync_unavail_sec"] / max(uf["sync_live_sec"], 1), 4)
            fresh_pct_a = round(100.0 * ua["n_unavail_fresh"] / max(ua["n_resolved_p0_fresh"], 1), 3)
            sync_pct_a = round(100.0 * ua["n_unavail_sync"] / max(ua["n_resolved_p0_sync"], 1), 3)

            gates_summary[str(k)] = {
                "k_days": k,
                "weighted_add_no_days_full": round(float(a_no_f.sum()), 2),
                "weighted_add_ghsa_days_full": round(float(a_gh_f.sum()), 2),
                "weighted_add_osv_days_full": round(float(a_os_f.sum()), 2),
                "weighted_add_zero_days_full": round(float(a_z0_f.sum()), 2),
                "weighted_r0_days_full": round(float((w * np.array([r["r0_days_full"] for r in recs])).sum()), 2),
                "rec_ghsa_full_pct": round(rec_gh_f, 2),
                "rec_ghsa_full_ci95": [round(rec_gh_f_lo, 2), round(rec_gh_f_hi, 2)],
                "rec_osv_full_pct": round(rec_os_f, 2),
                "rec_osv_full_ci95": [round(rec_os_f_lo, 2), round(rec_os_f_hi, 2)],
                "feed_gain_full_pp": round(feed_gain_f, 2),
                "feed_gain_full_ci95": [round(fg_lo, 2), round(fg_hi, 2)],
                "rec_zero_full_pct": round(rec_z0_f, 2),
                "rec_ghsa_act_pct": round(rec_gh_a, 2),
                "rec_ghsa_act_ci95": [round(rec_gh_a_lo, 2), round(rec_gh_a_hi, 2)],
                "rec_osv_act_pct": round(rec_os_a, 2),
                "rec_osv_act_ci95": [round(rec_os_a_lo, 2), round(rec_os_a_hi, 2)],
                "feed_gain_act_pp": round(feed_gain_a, 2),
                "feed_gain_act_ci95": [round(fga_lo, 2), round(fga_hi, 2)],
                "baseline_unw_ghsa_pct": round(b_unw_gh, 2),
                "baseline_unw_osv_pct": round(b_unw_os, 2),
                "gap_vs_unw_ghsa_pp": round(rec_gh_f - b_unw_gh, 2),
                "gap_vs_unw_ghsa_ci95": [round(gap_gh_lo, 2), round(gap_gh_hi, 2)],
                "gap_vs_unw_osv_pp": round(rec_os_f - b_unw_os, 2),
                "gap_vs_unw_osv_ci95": [round(gap_os_lo, 2), round(gap_os_hi, 2)],
                "rec_ghsa_recent_2022_pct": round(rec_gh_recent, 2),
                "rec_osv_recent_2022_pct": round(rec_os_recent, 2),
                "rec_ghsa_notop3_pct": round(rec_gh_notop3, 2),
                "rec_osv_notop3_pct": round(rec_os_notop3, 2),
                "per_rel_add_no_med_iqr_p90": [
                    round(float(np.median(per_adv_add_no)), 3),
                    round(float(np.percentile(per_adv_add_no, 25)), 3),
                    round(float(np.percentile(per_adv_add_no, 75)), 3),
                    round(float(np.percentile(per_adv_add_no, 90)), 3),
                ] if len(per_adv_add_no) else [0, 0, 0, 0],
                "per_rel_add_ghsa_med_iqr_p90": [
                    round(float(np.median(per_adv_add_gh)), 3),
                    round(float(np.percentile(per_adv_add_gh, 25)), 3),
                    round(float(np.percentile(per_adv_add_gh, 75)), 3),
                    round(float(np.percentile(per_adv_add_gh, 90)), 3),
                ] if len(per_adv_add_gh) else [0, 0, 0, 0],
                "unavail_fresh_pct_full": fresh_pct_f,
                "unavail_sync_pct_full": sync_pct_f,
                "unavail_fresh_day_pct_full": fresh_day_pct_f,
                "unavail_sync_day_pct_full": sync_day_pct_f,
                "unavail_fresh_pct_act": fresh_pct_a,
                "unavail_sync_pct_act": sync_pct_a,
            }

        # Bootstrap CI for fresh-install unavailability at k=7!
        u_cl = np.array([r["upstream"] for r in up_u])
        u_num = np.array([r.get("fresh_hit_7", 0) for r in up_u], dtype=np.float64)
        u_den = np.array([r.get("fresh_res0_7", 0) for r in up_u], dtype=np.float64)
        uf7_pt, uf7_lo, uf7_hi = eng.clustered_bootstrap_ci(u_cl, u_num, u_den, n_boot=n_boot, seed=SEED + 777, scale=100.0)
        gates_summary["7"]["unavail_fresh_pct_full_ci95"] = [round(uf7_lo, 3), round(uf7_hi, 3)]

        # Collect 4-cell openness values at k=7 for exact 3-factor Shapley decomposition!
        c_no_raw = np.array([sum(r["weight"] * r[f"c7_{fam}_no"] for r in recs) for fam in eng.OPENNESS_CLASSES])
        c_no = np.maximum(c_no_raw, 0.0)
        c_gh = np.array([sum(r["weight"] * r[f"c7_{fam}_ghsa"] for r in recs) for fam in eng.OPENNESS_CLASSES])
        c_os = np.array([sum(r["weight"] * r[f"c7_{fam}_osv"] for r in recs) for fam in eng.OPENNESS_CLASSES])
        tot_c_no = float(c_no.sum())
        w_cells = c_no / max(tot_c_no, 1e-12)
        nat_rec_cells = np.where(c_no > 0, np.clip((c_no - c_os) / np.maximum(c_no, 1e-12) * 100.0, 0.0, 100.0), 0.0)
        gh_rec_cells = np.where(c_no > 0, np.clip((c_no - c_gh) / np.maximum(c_no, 1e-12) * 100.0, 0.0, 100.0), 0.0)
        gh_loss_cells = np.maximum(0.0, nat_rec_cells - gh_rec_cells)

        # Conservative +12h end-of-day OSV sensitivity at k=7 using CRN (SEED + 7)
        a_no_7 = w * np.array([r["add_no_7_full"] for r in recs])
        a_gh_7 = w * np.array([r["add_ghsa_7_full"] for r in recs])
        a_oe_7 = w * np.array([r.get("add_osv_eod_7_full", r["add_osv_7_full"]) for r in recs])
        rec_osv_eod_7, roe_lo, roe_hi = eng.clustered_bootstrap_ci(
            cl_ids, a_no_7 - a_oe_7, a_no_7, n_boot=n_boot, seed=SEED + 7, scale=100.0
        )
        fg_eod_7, fge_lo, fge_hi = eng.clustered_bootstrap_ci(
            cl_ids, a_gh_7 - a_oe_7, a_no_7, n_boot=n_boot, seed=SEED + 7, scale=100.0
        )
        gates_summary["7"]["rec_osv_eod_full_pct"] = round(rec_osv_eod_7, 2)
        gates_summary["7"]["rec_osv_eod_full_ci95"] = [round(roe_lo, 2), round(roe_hi, 2)]
        gates_summary["7"]["feed_gain_eod_full_pp"] = round(fg_eod_7, 2)
        gates_summary["7"]["feed_gain_eod_full_ci95"] = [round(fge_lo, 2), round(fge_hi, 2)]
        cell_data_by_eco[eco] = {
            "w": w_cells,
            "nat_rec": nat_rec_cells,
            "gh_rec": gh_rec_cells,
            "gh_loss": gh_loss_cells,
            "c_no": c_no,
        }

        # Lag quantiles (days)
        lag_summary = {
            "ghsa_lag_days_p10_p25_med_p75_p90": [round(float(x) / eng.DAY, 2) for x in np.percentile(lag_ghsa, [10, 25, 50, 75, 90])] if len(lag_ghsa) else [],
            "osv_lag_days_p10_p25_med_p75_p90": [round(float(x) / eng.DAY, 2) for x in np.percentile(lag_osv, [10, 25, 50, 75, 90])] if len(lag_osv) else [],
            "ghsa_precedes_fix_pct": round(float(np.mean(lag_ghsa <= 0) * 100.0), 2) if len(lag_ghsa) else 0.0,
            "osv_precedes_fix_pct": round(float(np.mean(lag_osv <= 0) * 100.0), 2) if len(lag_osv) else 0.0,
            "vendor_lag_gt0_pct": round(float(np.mean(v_lag > 0) * 100.0), 2) if len(v_lag) else 0.0,
            "vendor_lag_gt7d_pct": round(float(np.mean(v_lag > 7 * eng.DAY) * 100.0), 2) if len(v_lag) else 0.0,
            "vendor_lag_med_days_pos": round(float(np.median(v_lag[v_lag > 0])) / eng.DAY, 2) if (v_lag > 0).any() else 0.0,
        }

        # Severity breakdown at k=7
        sev_breakdown = {}
        for sev in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
            sm = np.array([r["severity"] == sev for r in recs], dtype=bool)
            if sm.any():
                s_no = (w * np.array([r["add_no_7_full"] for r in recs]))[sm].sum()
                s_gh = (w * np.array([r["add_ghsa_7_full"] for r in recs]))[sm].sum()
                s_os = (w * np.array([r["add_osv_7_full"] for r in recs]))[sm].sum()
                sev_breakdown[sev] = {
                    "n_adv": int(sm.sum()),
                    "rec_ghsa_7_pct": round(float((s_no - s_gh) / max(s_no, 1e-12) * 100.0), 2),
                    "rec_osv_7_pct": round(float((s_no - s_os) / max(s_no, 1e-12) * 100.0), 2),
                }

        out_summary["ecosystems"][eco] = {
            "audit": er_res["audit"],
            "mal_summary": er_res["mal_summary"],
            "lag_summary": lag_summary,
            "top3_hubs": top3_ups,
            "top3_hub_share_pct": top3_share_pct,
            "gates": gates_summary,
            "severity_k7": sev_breakdown,
            "depth2_summary": er_res["depth2_summary"],
            "openness_k7_cells": {
                fam: {
                    "exposure_share_pct": round(float(w_cells[i] * 100.0), 2),
                    "nat_rec_pct": round(float(nat_rec_cells[i]), 2),
                    "ghsa_rec_pct": round(float(gh_rec_cells[i]), 2),
                    "ghsa_loss_pp": round(float(gh_loss_cells[i]), 2),
                }
                for i, fam in enumerate(eng.OPENNESS_CLASSES)
            },
        }

    # Exact 3-factor Average-Pairwise Shapley decomposition against the true equal-weighted 5-ecosystem mean!
    # Averaging pairwise Shapley(e vs r) across all r in ECOSYSTEMS guarantees:
    #   reference_val == mean_r target_val(r), sum_e Delta_e == 0, sum_e phi_X(e) == 0, additivity_err < 1e-12.
    shapley_table = {}
    for eco in ECOSYSTEMS:
        cd = cell_data_by_eco[eco]
        pair_res = []
        for ref_eco in ECOSYSTEMS:
            rcd = cell_data_by_eco[ref_eco]
            sh = eng.shapley_cell_recovery_model(
                cd["nat_rec"], cd["w"], rcd["nat_rec"], rcd["w"],
                cd["nat_rec"], cd["gh_loss"], rcd["nat_rec"], rcd["gh_loss"],
            )
            assert sh["additivity_err"] < 1e-9, f"Shapley additivity failed for {eco} vs {ref_eco}: {sh['additivity_err']}"
            pair_res.append(sh)
        avg_sh = {
            k: float(np.mean([p[k] for p in pair_res]))
            for k in ("phi_S", "phi_G", "phi_C", "total_gap", "reference_val", "target_val", "interact_SG", "interact_SC", "interact_GC")
        }
        avg_sh["additivity_err"] = abs((avg_sh["phi_S"] + avg_sh["phi_G"] + avg_sh["phi_C"]) - avg_sh["total_gap"])
        assert avg_sh["additivity_err"] < 1e-9, f"Average-pairwise Shapley additivity failed for {eco}: {avg_sh['additivity_err']}"
        shapley_table[eco] = {k: round(v, 4) for k, v in avg_sh.items()}

    # Alternate npm-anchored Shapley reference sensitivity!
    npm_cd = cell_data_by_eco["npm"]
    shapley_npm_ref = {}
    for eco in ECOSYSTEMS:
        cd = cell_data_by_eco[eco]
        sh = eng.shapley_cell_recovery_model(
            cd["nat_rec"], cd["w"], npm_cd["nat_rec"], npm_cd["w"], cd["nat_rec"], cd["gh_loss"], npm_cd["nat_rec"], npm_cd["gh_loss"]
        )
        assert sh["additivity_err"] < 1e-9
        shapley_npm_ref[eco] = {k: round(v, 4) for k, v in sh.items()}

    # Headline cross-ecosystem MME checks (K1, K2, K3)
    rec7_ghsa_vals = [out_summary["ecosystems"][e]["gates"]["7"]["rec_ghsa_full_pct"] for e in ECOSYSTEMS]
    rec7_osv_vals = [out_summary["ecosystems"][e]["gates"]["7"]["rec_osv_full_pct"] for e in ECOSYSTEMS]
    rec7_ghsa_act_vals = [out_summary["ecosystems"][e]["gates"]["7"]["rec_ghsa_act_pct"] for e in ECOSYSTEMS]
    unavail7_fresh_vals = [out_summary["ecosystems"][e]["gates"]["7"]["unavail_fresh_pct_full"] for e in ECOSYSTEMS]
    feed_gains_7 = {e: out_summary["ecosystems"][e]["gates"]["7"]["feed_gain_full_pp"] for e in ECOSYSTEMS}
    unw_gaps_7 = {e: out_summary["ecosystems"][e]["gates"]["7"]["gap_vs_unw_ghsa_pp"] for e in ECOSYSTEMS}

    out_summary["shapley_pooled_ref"] = shapley_table
    out_summary["shapley_npm_ref"] = shapley_npm_ref
    out_summary["headline_mme_checks"] = {
        "K1_rec7_ghsa_spread_full_pp": round(max(rec7_ghsa_vals) - min(rec7_ghsa_vals), 2),
        "K1_rec7_osv_spread_full_pp": round(max(rec7_osv_vals) - min(rec7_osv_vals), 2),
        "K1_rec7_ghsa_spread_act_pp": round(max(rec7_ghsa_act_vals) - min(rec7_ghsa_act_vals), 2),
        "K1_unavail7_fresh_spread_pp": round(max(unavail7_fresh_vals) - min(unavail7_fresh_vals), 3),
        "K1_mme_threshold_pp": 15.0,
        "K2_max_feed_gain_7_pp": max(feed_gains_7.values()),
        "K2_feed_gains_by_eco_pp": feed_gains_7,
        "K2_mme_threshold_pp": 5.0,
        "K3_max_abs_gap_vs_unw_7_pp": round(max(abs(v) for v in unw_gaps_7.values()), 2),
        "K3_gaps_vs_unw_by_eco_pp": unw_gaps_7,
        "K3_mme_threshold_pp": 10.0,
    }
    return out_summary


def export_audited_csvs(summary: dict, manual_samples: list[dict], prefix: str) -> list[str]:
    """Write audited CSV tables directly consumed by ve-tikz-v6 PGFPlots figures and LaTeX tables."""
    paths = []

    # 1. Main gate curve CSV (ecosystem x k)
    p1 = ART_DIR / f"{prefix}_gate_curves.csv"
    with p1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ecosystem", "k_days", "rec_ghsa_full", "rec_ghsa_lo", "rec_ghsa_hi",
            "rec_osv_full", "rec_osv_lo", "rec_osv_hi", "feed_gain_pp", "rec_zero_full",
            "rec_ghsa_act", "rec_osv_act", "baseline_unw_ghsa", "gap_vs_unw_ghsa",
            "unavail_fresh_pct", "unavail_sync_pct", "unavail_fresh_act_pct",
            "mal_block_all_pct", "mal_block_live2024_pct", "mal_block_live2025_pct",
            "mal_block_prev_legit_pct", "add_no_days_full", "add_ghsa_days_full",
        ])
        for eco in ECOSYSTEMS:
            ed = summary["ecosystems"][eco]
            ms = ed["mal_summary"]
            for k in eng.GATES:
                g = ed["gates"][str(k)]
                w.writerow([
                    eco, k, g["rec_ghsa_full_pct"], g["rec_ghsa_full_ci95"][0], g["rec_ghsa_full_ci95"][1],
                    g["rec_osv_full_pct"], g["rec_osv_full_ci95"][0], g["rec_osv_full_ci95"][1],
                    g["feed_gain_full_pp"], g["rec_zero_full_pct"],
                    g["rec_ghsa_act_pct"], g["rec_osv_act_pct"], g["baseline_unw_ghsa_pct"], g["gap_vs_unw_ghsa_pp"],
                    g["unavail_fresh_pct_full"], g["unavail_sync_pct_full"], g["unavail_fresh_pct_act"],
                    "" if ms["blocking_all_pct"][str(k)] is None else ms["blocking_all_pct"][str(k)],
                    "" if ms["blocking_live_2024_pct"][str(k)] is None else ms["blocking_live_2024_pct"][str(k)],
                    "" if ms["blocking_live_2025_pct"][str(k)] is None else ms["blocking_live_2025_pct"][str(k)],
                    "" if ms["blocking_prev_legit_pct"][str(k)] is None else ms["blocking_prev_legit_pct"][str(k)],
                    g["weighted_add_no_days_full"], g["weighted_add_ghsa_days_full"],
                ])
    paths.append(str(p1))

    # 2. Shapley decomposition CSV
    p2 = ART_DIR / f"{prefix}_shapley_k7.csv"
    with p2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ecosystem", "reference_val", "target_val", "total_gap", "phi_S_silent_lag", "phi_G_ghsa_curation", "phi_C_constraint_openness", "additivity_err"])
        for eco in ECOSYSTEMS:
            sh = summary["shapley_pooled_ref"][eco]
            w.writerow([eco, sh["reference_val"], sh["target_val"], sh["total_gap"], sh["phi_S"], sh["phi_G"], sh["phi_C"], sh["additivity_err"]])
    paths.append(str(p2))

    # 3. Manual audit sample CSV (100 records in official mode)
    if manual_samples:
        p3 = ART_DIR / "manual_audit_sample_100.csv"
        keys = list(manual_samples[0].keys())
        with p3.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(manual_samples)
        paths.append(str(p3))

    # 4. Openness x recovery cell breakdown at k=7 CSV
    p4 = ART_DIR / f"{prefix}_openness_k7.csv"
    with p4.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ecosystem", "openness_class", "exposure_share_pct", "nat_rec_pct", "ghsa_rec_pct", "ghsa_loss_pp"])
        for eco in ECOSYSTEMS:
            cells = summary["ecosystems"][eco]["openness_k7_cells"]
            for fam in eng.OPENNESS_CLASSES:
                c = cells[fam]
                w.writerow([eco, fam, c["exposure_share_pct"], c["nat_rec_pct"], c["ghsa_rec_pct"], c["ghsa_loss_pp"]])
    paths.append(str(p4))

    # 5. Sensitivity & robustness summary at k=7 CSV
    p5 = ART_DIR / f"{prefix}_sensitivity_k7.csv"
    with p5.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ecosystem", "rec_ghsa_full", "rec_ghsa_act", "rec_ghsa_recent_2022", "rec_ghsa_notop3",
            "rec_osv_full", "rec_osv_act", "rec_osv_recent_2022", "rec_osv_notop3",
            "baseline_unw_ghsa", "gap_vs_unw_ghsa", "top3_hub_share_pct",
            "unavail_fresh_full", "unavail_sync_full", "unavail_fresh_act", "unavail_sync_act",
        ])
        for eco in ECOSYSTEMS:
            ed = summary["ecosystems"][eco]
            g = ed["gates"]["7"]
            w.writerow([
                eco, g["rec_ghsa_full_pct"], g["rec_ghsa_act_pct"], g["rec_ghsa_recent_2022_pct"], g["rec_ghsa_notop3_pct"],
                g["rec_osv_full_pct"], g["rec_osv_act_pct"], g["rec_osv_recent_2022_pct"], g["rec_osv_notop3_pct"],
                g["baseline_unw_ghsa_pct"], g["gap_vs_unw_ghsa_pp"], ed["top3_hub_share_pct"],
                g["unavail_fresh_pct_full"], g["unavail_sync_pct_full"], g["unavail_fresh_pct_act"], g["unavail_sync_pct_act"],
            ])
    paths.append(str(p5))

    return paths


def _run_eco_worker(task: tuple[str, int, bool]) -> dict:
    import os as _os
    eco, cap, run_depth2 = task
    _os.environ["POLARS_MAX_THREADS"] = "2"
    adv_all_df = pl.read_parquet(DATA_ROOT / "advisories_census.parquet")
    mal_all_df = pl.read_parquet(DATA_ROOT / "malicious_packages.parquet")
    return run_ecosystem(eco, adv_all_df, mal_all_df, cap=cap, run_depth2=run_depth2)


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "official"])
    ap.add_argument("--cap", type=int, default=None, help="Override advisory cap (0 = 100%% complete census).")
    args = ap.parse_args()

    t_all = time.perf_counter()
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # Official mode runs the 100% complete census across all 5 ecosystems (cap=0, zero sampling variance!)
    cap = args.cap if args.cap is not None else (100 if args.mode == "pilot" else 0)
    n_boot = 500 if args.mode == "pilot" else 5000
    run_depth2 = args.mode == "official"

    tasks = [(eco, cap, run_depth2) for eco in ECOSYSTEMS]
    eco_results_by_name: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=len(ECOSYSTEMS)) as ex:
        for res in ex.map(_run_eco_worker, tasks):
            eco_results_by_name[res["ecosystem"]] = res

    eco_results = [eco_results_by_name[eco] for eco in ECOSYSTEMS]
    manual_samples = []
    for res in eco_results:
        manual_samples.extend(res["manual_sample"])

    summary = summarize_and_bootstrap(eco_results, mode=args.mode, n_boot=n_boot)
    summary["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 1)

    prefix = "census" if args.mode == "official" else f"census_{args.mode}"
    csv_paths = export_audited_csvs(summary, manual_samples, prefix)
    out_json = ART_DIR / f"{prefix}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Save SHA-256 receipt
    receipt = {
        "mode": args.mode,
        "generated_utc": summary["generated_utc"],
        "files": {
            p: {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "bytes": Path(p).stat().st_size}
            for p in [str(out_json)] + csv_paths
        },
    }
    rec_path = ART_DIR / f"{prefix}_receipt.json"
    rec_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print("\n================ HEADLINE MME SUMMARY ================", flush=True)
    print(json.dumps(summary["headline_mme_checks"], indent=2), flush=True)
    print(f"\nSaved {out_json} and {len(csv_paths)} CSVs in {summary['total_elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
