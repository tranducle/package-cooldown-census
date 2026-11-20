# Data Provenance, Access, and Licensing

All data analyzed in this study originate from public open-source metadata archives. No private enterprise lockfiles, personal user data, or proprietary vulnerability feeds are used.

## 1. Upstream Public Archives

1. **Registry Dependency Graphs (`ecosyste.ms` Packages Open Data):**
   - Snapshot date: February 5, 2026 (common observation cutoff `T_cut = 2026-02-05T00:00:00Z`)
   - Official source: https://packages.ecosyste.ms/open-data
   - License: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
   - Size note: Raw PostgreSQL `COPY` dumps (`packages.sql`, `versions.sql`, `dependencies.sql`) exceed `100 GB` uncompressed and are therefore referenced by upstream URL rather than committed to Git. Use `src/registry_parquet_indexer.py` to stream raw dumps directly into Zstandard-compressed Parquet partitions.

2. **Reviewed GitHub Advisory Database (GHSA):**
   - Snapshot commit: `674392f1` (retrieved November 10, 2026)
   - Official source: https://github.com/github/advisory-database
   - License: Creative Commons Attribution 4.0 (CC BY 4.0)

3. **Open Source Vulnerabilities (OSV) and OpenSSF Malicious Packages (`MAL-*`):**
   - Ecosystems: npm (retrieved November 10, 2026), PyPI, crates.io/Cargo, RubyGems, Packagist (retrieved November 19, 2026)
   - Official source: https://google.github.io/osv.dev/data/
   - License: Creative Commons Attribution 4.0 (CC BY 4.0)
   - Exact archive sizes, record counts, and SHA-256 receipts are recorded in `data/osv_archive_manifest.json`.

## 2. Bundled Public Derived Catalogs and Result Tables

To permit immediate verification without re-downloading `> 100 GB` of raw SQL dumps, this repository bundles the pre-cutoff unified advisory Parquet catalogs (`4.5 MB` combined) and aggregate census output tables, each hash-verified in `DATA_MANIFEST.json`:

| Path | Rows / Contents | Source License |
| --- | --- | --- |
| `data/advisories_census.parquet` | `22,807` deduplicated reviewed GHSA fixed advisories linked to native OSV timestamps | CC BY 4.0 |
| `data/malicious_packages.parquet` | `221,408` pre-cutoff OpenSSF `MAL-*` disclosure records (`t_OSV <= T_cut`) | CC BY 4.0 |
| `data/advisory_build_receipt.json` | Advisory merge counts, linkage statistics, and Parquet SHA-256 digests | CC BY 4.0 |
| `data/osv_archive_manifest.json` | Official OSV bulk ZIP sizes, SHA-256 digests, and prefix inventories | CC BY 4.0 |
| `results/census_summary.json` | Complete 100% five-ecosystem census outcomes, 5,000-replicate CRN bootstrap CIs, robustness slices, and exact Shapley shares | CC BY 4.0 |
| `results/census_gate_curves.csv` | Gate-by-gate exposure recovery, unavailability, and `MAL-*` coverage (`k in {0, 1, 3, 7, 14, 30}`) | CC BY 4.0 |
| `results/census_openness_k7.csv` | Seven-day exposure shares and recovery across constraint openness classes (`exact`, `patch`, `minor`, `wide`) | CC BY 4.0 |
| `results/census_sensitivity_k7.csv` | Multi-release, 2022 to 2026 era, top-3 hub deletion, and 50-advisory fan-out sensitivity metrics | CC BY 4.0 |
| `results/census_shapley_k7.csv` | Average-pairwise and npm-anchored exact three-factor Shapley accounting shares | CC BY 4.0 |
| `results/manual_audit_sample_100.csv` | 100-record stratified manual audit sample (20 advisories per ecosystem) | CC BY 4.0 |
| `results/census_run_register.csv` | Top-level census verification register | CC BY 4.0 |
| `results/corpus_audit_metrics.json` | Pre-replay manifest checks and 5,000-constraint live parse coverage audit | CC BY 4.0 |
| `results/census_receipt.json` | SHA-256 digest receipt for core census output files | CC BY 4.0 |
| `results/figure_data/*.csv` | Five figure-ready CSV tables regenerated deterministically from `results/census_summary.json` | CC BY 4.0 |

## 3. Access Restrictions

None. Every upstream archive and bundled derived file is publicly accessible under open licenses (CC BY 4.0 / CC BY-SA 4.0 for data; MIT License for software).
