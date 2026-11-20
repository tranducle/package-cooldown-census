# Reproducibility Guide

## 1. Software Environment

All experiments and verification checks were executed on Python **3.12.10** with pinned dependencies in `requirements.txt`:
- `polars==1.42.1`
- `pyarrow==23.0.1`
- `numpy==2.3.5`
- `packaging==26.2`
- `scipy==1.17.1`

Install into a clean virtual environment:

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Deterministic Offline Verification (< 5 Seconds)

Run the self-contained verification entrypoint from the repository root:

```bash
python scripts/verify_reproducibility.py --mode verify
```

This command executes three checks:
1. Runs all 14 tests in `tests/test_ecosystem_matchers.py` and `tests/test_cooldown_replay_engine.py`.
2. Re-runs `scripts/generate_figure_data.py` against `results/census_summary.json` into a temporary directory and asserts byte-exact SHA-256 parity with all five CSV files in `results/figure_data/`.
3. Confirms zero added exposure and zero unavailability at `k=0`, zero feed shift in npm and RubyGems, `n_boot == 5000`, zero-sum cross-ecosystem Shapley deltas, and exact Shapley additivity (`< 1e-9`).

## 3. Full End-to-End Archival Replay (Optional)

Rebuilding all intermediate tables from raw public upstream dumps requires approximately `120 GB` of temporary local disk space and took `26.4` minutes across five parallel ecosystem workers for the census replay:

1. Download the February 5, 2026 `ecosyste.ms` Packages Open Data dump (`https://packages.ecosyste.ms/open-data`) and stream-index the five registries into Parquet partitions:
   ```bash
   python src/registry_parquet_indexer.py
   ```
2. Download official OSV bulk archives and reviewed GHSA commit `674392f1`, then build the unified pre-cutoff advisory Parquet catalogs:
   ```bash
   python scripts/fetch_osv_archives.py
   python src/advisory_catalog_builder.py
   ```
3. Run the manifest, corpus integrity, and 5,000-constraint parse audit:
   ```bash
   python experiments/run_corpus_audit.py
   ```
4. Execute the complete 100% five-ecosystem census replay (fixed bootstrap seed `20261119`):
   ```bash
   python experiments/run_census_replay.py --mode official
   ```
5. Regenerate hashed figure CSV tables:
   ```bash
   python scripts/generate_figure_data.py
   ```

Environment variable overrides (`COOLDOWN_DATA_ROOT`, `COOLDOWN_ADV_DIR`, `COOLDOWN_DATA_DIR`, `COOLDOWN_RESULTS_DIR`, `COOLDOWN_SUMMARY_PATH`, `COOLDOWN_FIGDATA_DIR`) allow storing raw archives outside the repository directory.
