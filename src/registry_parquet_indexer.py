"""Outcome-blind multi-ecosystem streaming COPY-to-Parquet indexer for ecosyste.ms 2026-02-05.

Extracts PyPI (registry_id=11), Cargo (registry_id=2), RubyGems (registry_id=12), and
Packagist (registry_id=9) from the local SHA-256 verified SQL COPY dumps in one pass per
table, writing SHA-256 verified ZSTD Parquet files plus a complete provenance manifest.
Never reads any advisory or policy outcome. Existing npm Parquet index is untouched.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("COOLDOWN_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data")))
TABLES_DIR = DATA_ROOT / "ecosystems_open_data" / "tables"
OUT_DIR = DATA_ROOT / "ecosystems_open_data" / "index_multieco"

REG_TO_ECO = {
    b"11": "pypi",
    b"2": "cargo",
    b"12": "rubygems",
    b"9": "packagist",
}
ECO_BYTES = {
    b"pypi": "pypi",
    b"cargo": "cargo",
    b"rubygems": "rubygems",
    b"packagist": "packagist",
}
# Exact non-development runtime dependency kinds verified from dependencies.sql schema probe
KEEP_KINDS = {
    "pypi": {b"runtime"},
    "cargo": {b"normal"},
    "rubygems": {b"runtime"},
    "packagist": {b"runtime"},
}

COPY_RE = re.compile(rb"\nCOPY public\.(\w+) \(([^)]*)\) FROM stdin;\n")
NULL = b"\\N"
ESC = re.compile(rb"\\(?:([0-7]{1,3})|x([0-9A-Fa-f]{1,2})|(.))", re.S)
SIMPLE = {b"b": b"\b", b"f": b"\f", b"n": b"\n", b"r": b"\r", b"t": b"\t", b"v": b"\v"}
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def unescape(v: bytes) -> str:
    if b"\\" not in v:
        return v.decode("utf-8", "replace")

    def f(m):
        if m.group(1):
            return bytes([int(m.group(1), 8) & 0xFF])
        if m.group(2):
            return bytes([int(m.group(2), 16)])
        c = m.group(3)
        return SIMPLE.get(c, c)

    return ESC.sub(f, v).decode("utf-8", "replace")


def ts_us(v: bytes) -> int:
    if v == NULL or not v:
        return -1
    s = v.decode("ascii", "replace")
    try:
        d = dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        frac = s[20:26] if len(s) > 19 else ""
        us = int((frac + "000000")[:6]) if frac.isdigit() else 0
        return int((d - EPOCH).total_seconds()) * 1_000_000 + us
    except ValueError:
        return -1


_PID_SORTED: np.ndarray | None = None
_PID_ECO_CODE: np.ndarray | None = None
ECO_LIST = ["pypi", "cargo", "rubygems", "packagist"]


def _load_pid_map(path_ids: str, path_codes: str):
    global _PID_SORTED, _PID_ECO_CODE
    if _PID_SORTED is None:
        _PID_SORTED = np.load(path_ids)
        _PID_ECO_CODE = np.load(path_codes)
    return _PID_SORTED, _PID_ECO_CODE


def work_packages(job):
    bi, block, idx, out_root = job
    lines = block.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    c_id, c_reg, c_name, c_status = idx["id"], idx["registry_id"], idx["name"], idx["status"]
    buckets = {e: ([], [], []) for e in ECO_LIST}
    for ln in lines:
        if not ln:
            continue
        f = ln.split(b"\t")
        if len(f) <= max(c_id, c_reg, c_name, c_status):
            continue
        eco = REG_TO_ECO.get(f[c_reg])
        if eco is None:
            continue
        ids, names, statuses = buckets[eco]
        ids.append(int(f[c_id]))
        names.append(unescape(f[c_name]))
        statuses.append(None if f[c_status] == NULL else unescape(f[c_status]))
    parts = {}
    for eco, (ids, names, statuses) in buckets.items():
        if not ids:
            continue
        t = pa.table(
            {
                "id": pa.array(ids, pa.int64()),
                "name": pa.array(names, pa.string()),
                "status": pa.array(statuses, pa.string()),
            }
        )
        p = Path(out_root) / "packages" / eco / f"part-{bi:05d}.parquet"
        pq.write_table(t, p, compression="zstd")
        parts[eco] = {
            "file": p.name,
            "rows_kept": len(ids),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    return {"block": bi, "rows_seen": len(lines), "parts": parts}


def work_versions(job):
    bi, block, idx, ids_path, codes_path, out_root = job
    pids_sorted, eco_codes = _load_pid_map(ids_path, codes_path)
    lines = block.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    c_id, c_pid, c_num, c_pub, c_reg = (
        idx["id"],
        idx["package_id"],
        idx["number"],
        idx["published_at"],
        idx["registry_id"],
    )
    n = len(lines)
    if n == 0:
        return {"block": bi, "rows_seen": 0, "parts": {}}
    raw_pids = np.empty(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for i, ln in enumerate(lines):
        parts_line = ln.split(b"\t", c_pid + 1)
        if len(parts_line) <= c_pid:
            raw_pids[i] = -1
            valid[i] = False
        else:
            raw_pids[i] = int(parts_line[c_pid])
    pos = np.searchsorted(pids_sorted, raw_pids)
    in_bounds = (pos < len(pids_sorted)) & valid
    matched = np.zeros(n, dtype=bool)
    matched[in_bounds] = pids_sorted[pos[in_bounds]] == raw_pids[in_bounds]

    buckets = {e: ([], [], [], [], []) for e in ECO_LIST}
    matched_indices = np.flatnonzero(matched)
    for i in matched_indices:
        eco = ECO_LIST[int(eco_codes[pos[i]])]
        f = lines[i].split(b"\t")
        if len(f) <= max(c_id, c_pid, c_num, c_pub, c_reg):
            continue
        ids, pids, nums, pubs, regs = buckets[eco]
        ids.append(int(f[c_id]))
        pids.append(int(f[c_pid]))
        nums.append(unescape(f[c_num]))
        pubs.append(ts_us(f[c_pub]))
        regs.append(-1 if f[c_reg] == NULL else int(f[c_reg]))

    parts = {}
    for eco, (ids, pids, nums, pubs, regs) in buckets.items():
        if not ids:
            continue
        t = pa.table(
            {
                "id": pa.array(ids, pa.int64()),
                "package_id": pa.array(pids, pa.int64()),
                "number": pa.array(nums, pa.string()),
                "published_at_us": pa.array(pubs, pa.int64()),
                "registry_id": pa.array(regs, pa.int64()),
            }
        )
        p = Path(out_root) / "versions" / eco / f"part-{bi:05d}.parquet"
        pq.write_table(t, p, compression="zstd")
        parts[eco] = {
            "file": p.name,
            "rows_kept": len(ids),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    return {"block": bi, "rows_seen": n, "parts": parts}


def work_dependencies(job):
    bi, block, idx, out_root = job
    lines = block.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    c_vid, c_name, c_eco, c_kind, c_opt, c_req = (
        idx["version_id"],
        idx["package_name"],
        idx["ecosystem"],
        idx["kind"],
        idx["optional"],
        idx["requirements"],
    )
    max_col = max(c_vid, c_name, c_eco, c_kind, c_opt, c_req)
    buckets = {e: ([], [], [], [], []) for e in ECO_LIST}
    dropped = {e: defaultdict(int) for e in ECO_LIST}
    for ln in lines:
        if not ln:
            continue
        f = ln.split(b"\t")
        if len(f) <= max_col:
            continue
        eco = ECO_BYTES.get(f[c_eco])
        if eco is None:
            continue
        k = f[c_kind]
        if k not in KEEP_KINDS[eco]:
            kstr = k.decode("utf-8", "replace")
            dropped[eco][kstr] += 1
            continue
        vids, names, kinds, reqs, opts = buckets[eco]
        vids.append(-1 if f[c_vid] == NULL else int(f[c_vid]))
        names.append(unescape(f[c_name]))
        kinds.append(k.decode("utf-8", "replace"))
        reqs.append(None if f[c_req] == NULL else unescape(f[c_req]))
        opts.append(None if f[c_opt] == NULL else f[c_opt] == b"t")

    parts = {}
    for eco, (vids, names, kinds, reqs, opts) in buckets.items():
        if not vids:
            continue
        t = pa.table(
            {
                "version_id": pa.array(vids, pa.int64()),
                "package_name": pa.array(names, pa.string()),
                "kind": pa.array(kinds, pa.string()),
                "requirements": pa.array(reqs, pa.string()),
                "optional": pa.array(opts, pa.bool_()),
            }
        )
        p = Path(out_root) / "dependencies" / eco / f"part-{bi:05d}.parquet"
        pq.write_table(t, p, compression="zstd")
        parts[eco] = {
            "file": p.name,
            "rows_kept": len(vids),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    return {
        "block": bi,
        "rows_seen": len(lines),
        "parts": parts,
        "dropped": {e: dict(d) for e, d in dropped.items() if d},
    }


def stream_table(table_name: str, sql_path: Path, worker_fn, extra_args: tuple, workers: int = 7, block_mb: int = 64):
    t0 = time.perf_counter()
    for eco in ECO_LIST:
        d = OUT_DIR / table_name / eco
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    blk = block_mb << 20
    header = b""
    carry = b""
    idx = None
    state = "header"
    bi = 0
    terminated = False
    rows_seen_total = 0
    parts_by_eco: dict[str, list[dict]] = {e: [] for e in ECO_LIST}
    dropped_by_eco: dict[str, dict[str, int]] = {e: defaultdict(int) for e in ECO_LIST}
    inflight: set = set()

    def collect_done(futs):
        nonlocal rows_seen_total
        for fut in futs:
            r = fut.result()
            rows_seen_total += r["rows_seen"]
            for eco, pinfo in r.get("parts", {}).items():
                parts_by_eco[eco].append(pinfo)
            for eco, dmap in r.get("dropped", {}).items():
                for k, v in dmap.items():
                    dropped_by_eco[eco][k] += v

    with cf.ProcessPoolExecutor(max_workers=workers) as ex, sql_path.open("rb") as fh:
        while not terminated:
            chunk = fh.read(blk)
            if not chunk:
                break
            if state == "header":
                header += chunk
                m = COPY_RE.search(header)
                if not m:
                    continue
                cols = [c.strip().strip('"') for c in m.group(2).decode().split(",")]
                idx = {c: i for i, c in enumerate(cols)}
                chunk = header[m.end():]
                header = b""
                state = "data"
            data = carry + chunk
            term_pos = data.find(b"\n\\.\n")
            if term_pos != -1:
                block = data[: term_pos + 1]
                carry = b""
                terminated = True
            elif data.startswith(b"\\.\n"):
                block = b""
                carry = b""
                terminated = True
            else:
                nl = data.rfind(b"\n")
                if nl == -1:
                    carry = data
                    continue
                block = data[: nl + 1]
                carry = data[nl + 1 :]
            if block:
                inflight.add(ex.submit(worker_fn, (bi, block, idx, *extra_args, str(OUT_DIR))))
                bi += 1
                if len(inflight) >= workers * 2:
                    done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                    collect_done(done)
        if carry and not terminated:
            inflight.add(ex.submit(worker_fn, (bi, carry, idx, *extra_args, str(OUT_DIR))))
        if inflight:
            done, _ = cf.wait(inflight)
            collect_done(done)

    for eco in ECO_LIST:
        parts_by_eco[eco].sort(key=lambda x: x["file"])

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "table": table_name,
        "source_file": str(sql_path),
        "source_bytes": sql_path.stat().st_size,
        "copy_terminated": terminated,
        "rows_seen": rows_seen_total,
        "elapsed_seconds": elapsed,
        "ecosystems": {
            eco: {
                "parts_count": len(parts_by_eco[eco]),
                "rows_kept": sum(p["rows_kept"] for p in parts_by_eco[eco]),
                "dropped_kinds": dict(dropped_by_eco[eco]),
                "parts": parts_by_eco[eco],
            }
            for eco in ECO_LIST
        },
    }
    manifest_path = OUT_DIR / f"{table_name}.MANIFEST.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{table_name}] DONE in {elapsed}s | rows_seen={rows_seen_total:,} | "
        + ", ".join(f"{e}={summary['ecosystems'][e]['rows_kept']:,}" for e in ECO_LIST),
        flush=True,
    )
    return summary


def build_pid_arrays() -> tuple[Path, Path]:
    all_ids = []
    all_codes = []
    for code, eco in enumerate(ECO_LIST):
        eco_dir = OUT_DIR / "packages" / eco
        parts = sorted(eco_dir.glob("part-*.parquet"))
        for p in parts:
            t = pq.read_table(p, columns=["id"])
            arr = t["id"].to_numpy()
            all_ids.append(arr)
            all_codes.append(np.full(len(arr), code, dtype=np.uint8))
    ids = np.concatenate(all_ids)
    codes = np.concatenate(all_codes)
    order = np.argsort(ids)
    ids = ids[order]
    codes = codes[order]
    tmp_dir = OUT_DIR / "_ids"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p_ids = tmp_dir / "multieco_package_ids.npy"
    p_codes = tmp_dir / "multieco_package_eco_codes.npy"
    np.save(p_ids, ids)
    np.save(p_codes, codes)
    print(f"[pid_arrays] Saved {len(ids):,} sorted package_ids across 4 ecosystems", flush=True)
    return p_ids, p_codes


def main() -> None:
    t_all = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stream_table("packages", TABLES_DIR / "packages.sql", work_packages, ())
    p_ids, p_codes = build_pid_arrays()
    stream_table(
        "versions",
        TABLES_DIR / "versions.sql",
        work_versions,
        (str(p_ids), str(p_codes)),
    )
    stream_table("dependencies", TABLES_DIR / "dependencies.sql", work_dependencies, ())
    print(f"[ALL] Multi-ecosystem indexing complete in {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
