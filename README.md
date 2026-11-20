# Package Cooldown Census

Replication code, unit and semantic-mutation tests, pre-cutoff advisory catalogs, and aggregate census tables for the study:

> **Release-Age Cooldowns, Security Exemptions, and Advisory-Feed Lag Across Five Package Ecosystems: An Archival Census**
> Truong Duy Dinh, Tran Duc Le, and Cong Danh Nguyen (2026)

## Scope

This repository provides a deterministic archival replay pipeline evaluating release-age cooldown controls (`k` in `{0, 1, 3, 7, 14, 30}` days) across direct runtime dependency layers in five range-based package registries through February 5, 2026 (`2026-02-05T00:00:00Z`): **npm**, **PyPI**, **Cargo**, **RubyGems**, and **Packagist**.

The pipeline implements:
- Validated single-edge version-constraint matchers (`src/npm_semver.py`, `src/ecosystem_matchers.py`).
- Continuous-time counterfactual exposure integration, fresh-install versus shift-invariant synchronized unavailability, common-random-numbers (CRN) upstream-package clustered bootstraps (`5,000` replicates, seed `20261119`), and exact average-pairwise three-factor Shapley accounting (`src/cooldown_replay_engine.py`).
- Multi-feed advisory catalog ingestion combining reviewed GitHub Security Advisories (GHSA), ecosystem-native Open Source Vulnerabilities (OSV) feeds (`PYSEC`, `RUSTSEC`, `DRUPAL`), and OpenSSF malicious-package (`MAL-*`) records (`src/advisory_catalog_builder.py`).
- Streaming PostgreSQL `COPY` indexing for the public `ecosyste.ms` Packages Open Data archive (`src/registry_parquet_indexer.py`).

## Quick Offline Verification (< 5 Seconds)

All unit tests, figure-data SHA-256 checks, deterministic `k=0` / no-native-feed invariants, and exact Shapley additivity checks run offline using the bundled summary artifacts without downloading raw multi-gigabyte dumps:

```bash
python -m pip install -r requirements.txt
python scripts/verify_reproducibility.py --mode verify
```

Expected output:
1. All **14** unit, boundary, and semantic-mutation tests pass (`tests/test_ecosystem_matchers.py` and `tests/test_cooldown_replay_engine.py`).
2. Regenerated CSV tables under a temporary directory match `results/figure_data/*.csv` byte-for-byte by SHA-256.
3. All `k=0` and no-native-feed invariants equal `0.0`, bootstrap replicates equal `5,000`, and exact average-pairwise Shapley additivity error is `0.00e+00`.

## Repository Layout

```text
README.md                        Overview, quick verification, citation
DATA.md                          Upstream public archives, schemas, SHA-256 provenance, licenses
REPRODUCIBILITY.md               Environment specification and full end-to-end replay steps
DATA_MANIFEST.json               Hash-bound authorization manifest for bundled data and result tables
RELEASE_SHA256.txt               SHA-256 checksums for all tracked repository files
requirements.txt                 Pinned Python dependencies (tested on Python 3.12.10)
LICENSE                          MIT License (code) and CC BY 4.0 / CC BY-SA 4.0 notice (derived tables)

src/
  npm_semver.py                  Port of node-semver 7.8.6 range and version semantics
  ecosystem_matchers.py          Five-ecosystem single-edge matchers and openness classifier
  cooldown_replay_engine.py      Continuous-time exposure replay, CRN bootstraps, Shapley accounting
  advisory_catalog_builder.py    GHSA, native OSV, and OpenSSF MAL-* catalog merger
  registry_parquet_indexer.py    Streaming ecosyste.ms PostgreSQL COPY to Parquet indexer

experiments/
  run_census_replay.py           Full 100% five-ecosystem census replay runner
  run_corpus_audit.py            Index manifest, advisory corpus, and 5,000-constraint parse audit

scripts/
  verify_reproducibility.py      Offline verification suite and full-replay launcher
  generate_figure_data.py        Deterministic CSV generator from results/census_summary.json
  fetch_osv_archives.py          Official OSV bulk archive downloader with SHA-256 receipts

tests/
  test_ecosystem_matchers.py     9 matcher semantics, canonicalization, and mutation tests
  test_cooldown_replay_engine.py 5 replay, shift-invariance, CRN bootstrap, and Shapley tests

data/                            Bundled pre-cutoff public advisory catalogs and build manifests
results/                         Complete 100% census summary JSON, CSV tables, and audit logs
docs/                            100-record stratified manual audit summary
```

## Headline Census Metrics (Verified Against `results/census_summary.json`)

- **Census scale:** `7,025,881` packages, `80.69M` versions, `22,807` deduplicated reviewed GHSA fixed advisories, `11,940` eligible fixed advisories (`4,158` unique upstreams), `9,887` evaluated advisories (`3,352` unique upstreams, `74,838,411` parsed runtime declarations), and `211,457` valid pre-cutoff `MAL-*` matches.
- **Seven-day exemption recovery spread:** `10.58 pp` under GHSA-only exemptions (`45.45%` Packagist to `56.03%` RubyGems) versus `38.59 pp` under earliest native OSV triggers (`48.04%` Packagist to `86.63%` Cargo).
- **Cargo native `RUSTSEC` gain:** `+37.52 pp` (`95%` CI `[25.92, 50.65]`; `+33.82 pp` under the conservative `+12h` UTC end-of-day bound), reducing absolute waiting exposure `55.2%` below three-day GHSA gating.
- **Unavailability at `k=7` days:** `3.57`-fold asymmetric fresh-install spread (`0.95%` PyPI to `3.38%` RubyGems) collapsing to `<= 0.24%` under synchronized cooldowns.

## Citation and Contact

If you reuse this code or dataset, cite the accompanying manuscript (*Release-Age Cooldowns, Security Exemptions, and Advisory-Feed Lag Across Five Package Ecosystems: An Archival Census*, Truong Duy Dinh, Tran Duc Le, and Cong Danh Nguyen, 2026). For correspondence, contact Tran Duc Le (`let@uwstout.edu`).
