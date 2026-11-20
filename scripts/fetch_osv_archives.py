"""Download official OSV GCS archives for PyPI, crates.io, RubyGems, Packagist with SHA-256 receipts
and inventory native vs GHSA vs MAL-* advisory counts across all 5 ecosystems (strictly outcome-blind counts)."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ADV_DIR = Path(os.environ.get("COOLDOWN_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data"))) / "advisories"
TARGETS = {
    "pypi": "https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip",
    "cargo": "https://osv-vulnerabilities.storage.googleapis.com/crates.io/all.zip",
    "rubygems": "https://osv-vulnerabilities.storage.googleapis.com/RubyGems/all.zip",
    "packagist": "https://osv-vulnerabilities.storage.googleapis.com/Packagist/all.zip",
}


def download_with_hash(url: str, dest: Path) -> dict:
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "package-cooldown-census/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    h = hashlib.sha256()
    size = 0
    with dest.open("rb") as fh:
        while True:
            b = fh.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
            size += len(b)
    return {
        "url": url,
        "file": str(dest),
        "bytes": size,
        "sha256": h.hexdigest(),
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def inventory_zip(zpath: Path) -> dict:
    prefixes = Counter()
    with_fix = Counter()
    with_pub = Counter()
    total = 0
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            total += 1
            d = json.loads(zf.read(name))
            oid = str(d.get("id") or "")
            pref = oid.split("-", 1)[0] if "-" in oid else "OTHER"
            prefixes[pref] += 1
            if d.get("published"):
                with_pub[pref] += 1
            has_fix = False
            for aff in d.get("affected") or []:
                for r in aff.get("ranges") or []:
                    if any("fixed" in ev for ev in (r.get("events") or [])):
                        has_fix = True
                        break
            if has_fix:
                with_fix[pref] += 1
    return {
        "total_records": total,
        "prefixes": dict(prefixes.most_common(10)),
        "with_fixed": dict(with_fix.most_common(10)),
        "with_published_ts": dict(with_pub.most_common(10)),
    }


def main() -> None:
    receipts = {}
    for eco, url in TARGETS.items():
        dest = ADV_DIR / f"osv_{eco}_all.zip"
        info = download_with_hash(url, dest)
        inv = inventory_zip(dest)
        info["inventory"] = inv
        receipts[eco] = info
        print(f"[{eco}] bytes={info['bytes']:,} total={inv['total_records']:,} prefixes={inv['prefixes']} fixed={inv['with_fixed']}", flush=True)

    # Also inventory existing npm OSV zip!
    npm_zip = ADV_DIR / "osv_npm_all.zip"
    if npm_zip.exists():
        inv_npm = inventory_zip(npm_zip)
        receipts["npm"] = {
            "file": str(npm_zip),
            "bytes": npm_zip.stat().st_size,
            "inventory": inv_npm,
        }
        print(f"[npm] existing bytes={npm_zip.stat().st_size:,} total={inv_npm['total_records']:,} prefixes={inv_npm['prefixes']} fixed={inv_npm['with_fixed']}", flush=True)

    out_manifest = ADV_DIR / "osv_archive_manifest.json"
    out_manifest.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_manifest}")


if __name__ == "__main__":
    main()
