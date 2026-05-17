# MIA Climate ERA5-Land Pipeline

A script-based ERA5-Land climate processing pipeline for region-first
acquisition planning, local preprocessing, climate-index generation, and
validation.

This repository reimplements the useful part of a legacy climate workflow
as explicit, reviewable pipeline artifacts. It is intentionally **not** a
published Python package. The project is a small set of command-line
scripts plus shared helpers in `lib/`.

## What This Pipeline Does

The pipeline starts from a region polygon and builds a reproducible chain:

1. Validate the region geometry.
2. Plan ERA5-Land downloads without touching the network.
3. Optionally acquire ERA5-Land NetCDF files through the Copernicus CDS API.
4. Preprocess raw NetCDF files into daily standard products.
5. Compute annual climate indices.
6. Validate manifests and product schemas.
7. Run and audit a tightly scoped live smoke test before scaling up.

Current development uses the canonical case-study polygon:

```text
01_data/case_studies/rbmn.geojson
```

This is the Marismas Nacionales Nayarit region polygon used for local tests
and reference manifests.

## Current Scope

In scope:

- ERA5-Land daily statistics:
  - `tmax`
  - `tmin`
  - `tmean`
  - `u10m`
  - `v10m`
- ERA5-Land hourly precipitation:
  - `tp` -> daily `pr`
- Annual temperature indices:
  - `Tmx`, `Tmn`, `TXx`, `TNn`, `DTR`, `SU`, `TR`
- Annual precipitation indices:
  - `PRCPTOT`, `RX1day`, `R95p`, `CDD`, `CWD`, `R10mm`, `R20mm`
- Dry-run orchestration, validation, live-smoke readiness, and live-smoke
  audit tooling.

Paused / out of current scope:

- CHIRPS
- Daymet
- Livneh
- WRF extraction / merge snippets
- CHELSA, which does not appear in the reviewed legacy material
- Docker, Google Cloud Storage, and Cloud Run execution
- full 175-request live ERA5-Land execution
- legacy-output numeric regression against archived NetCDF products
- wind-derived indices and percentile / spell-duration temperature indices

The legacy repository contains small CHIRPS, Daymet, Livneh, and WRF
snippets, but the complete Python workflow being reimplemented here is
ERA5-Land.

## Repository Layout

```text
configs/
  rbmn_local.json

01_data/
  case_studies/
    rbmn.geojson
  data_sources.md
  storage_layout.md

lib/
  regions.py
  download_plan.py
  acquisition.py
  preprocessing.py
  precipitation.py
  indices_temperature.py
  indices_precipitation.py
  index_manifest.py
  validation.py
  pipeline_runner.py
  live_smoke.py
  live_smoke_audit.py

scripts/
  00_validate_region.py
  01_plan_downloads.py
  02_download_era5_land.py
  03_preprocess_daily.py
  04_compute_indices.py
  05_run_pipeline.py
  06_validate_outputs.py
  07_run_live_smoke.py
  08_audit_live_smoke.py

runs/dev_region/
  *.json reference manifests

tests/
  pytest suite with synthetic fixtures
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install development and preprocessing dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,preprocessing]"
```

For live CDS acquisition only, install the acquisition extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[acquisition]"
```

Live acquisition also requires Copernicus CDS credentials outside the repo,
for example in `~/.cdsapirc` or through `CDSAPI_URL` / `CDSAPI_KEY`.

## Run The Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The tests use dry-run manifests and synthetic NetCDF fixtures. They do not
call CDS and do not require credentials.

## Dry-Run The Local Pipeline

The safest full local check is the dry-run pipeline runner:

```powershell
.\.venv\Scripts\python.exe scripts\05_run_pipeline.py `
    --config configs/rbmn_local.json `
    --mode dry-run `
    --output runs/dev_region/pipeline_manifest.json
```

Then validate the dry-run artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\06_validate_outputs.py `
    --pipeline-manifest runs/dev_region/pipeline_manifest.json `
    --output runs/dev_region/validation_report.json `
    --output-root runs/dev_region `
    --mode dry-run
```

Dry-run mode writes JSON manifests only. It should not create raw NetCDF,
intermediate products, or derived products under `runs/dev_region/`.

## Main Script Stages

### 1. Validate Region

```powershell
.\.venv\Scripts\python.exe scripts\00_validate_region.py `
    --region-id rbmn `
    --geometry 01_data/case_studies/rbmn.geojson `
    --output runs/dev_region/region_manifest.json
```

Produces `region_manifest.json` with CRS, geometry hash, and bounding box.

### 2. Plan Downloads

```powershell
.\.venv\Scripts\python.exe scripts\01_plan_downloads.py `
    --region-manifest runs/dev_region/region_manifest.json `
    --output runs/dev_region/download_manifest.json `
    --start-year 2000 `
    --end-year 2024
```

Produces a planned-only ERA5-Land request manifest:

- 125 daily-statistics requests
- 50 hourly-precipitation requests

### 3. Acquisition Adapter

Dry-run:

```powershell
.\.venv\Scripts\python.exe scripts\02_download_era5_land.py `
    --download-manifest runs/dev_region/download_manifest.json `
    --output runs/dev_region/acquisition_manifest.json `
    --output-root runs/dev_region `
    --mode dry-run `
    --limit 3
```

Execute mode exists, but should be used only through the live-smoke path
until the first CDS request has been audited.

### 4. Preprocess Daily Products

Dry-run daily statistics:

```powershell
.\.venv\Scripts\python.exe scripts\03_preprocess_daily.py `
    --acquisition-manifest runs/dev_region/acquisition_manifest.json `
    --download-manifest runs/dev_region/download_manifest.json `
    --region-manifest runs/dev_region/region_manifest.json `
    --output runs/dev_region/preprocessing_manifest.json `
    --output-root runs/dev_region `
    --mode dry-run
```

Precipitation is policy-gated:

```powershell
.\.venv\Scripts\python.exe scripts\03_preprocess_daily.py `
    --acquisition-manifest runs/dev_region/acquisition_manifest_precipitation_dry_run.json `
    --download-manifest runs/dev_region/download_manifest.json `
    --region-manifest runs/dev_region/region_manifest.json `
    --output runs/dev_region/preprocessing_manifest_precipitation.json `
    --output-root runs/dev_region `
    --mode dry-run `
    --precipitation-policy legacy_utc_minus_7
```

The `legacy_utc_minus_7` policy is a compatibility policy that matches the
legacy fixed-offset intent. The final scientific choice of UTC vs fixed
UTC-7 vs region-specific civil time remains open.

### 5. Compute Indices

Temperature indices:

```powershell
.\.venv\Scripts\python.exe scripts\04_compute_indices.py `
    --preprocessing-manifest runs/dev_region/preprocessing_manifest.json `
    --output runs/dev_region/index_manifest.json `
    --output-root runs/dev_region `
    --mode dry-run
```

Precipitation indices:

```powershell
.\.venv\Scripts\python.exe scripts\04_compute_indices.py `
    --preprocessing-manifest runs/dev_region/preprocessing_manifest_precipitation.json `
    --output runs/dev_region/index_manifest_precipitation.json `
    --output-root runs/dev_region `
    --mode dry-run `
    --index-family precipitation
```

## Live Smoke Test

Before scaling to all requests, run exactly one owner-authorized live CDS
request:

```powershell
.\.venv\Scripts\python.exe scripts\07_run_live_smoke.py `
    --config configs/rbmn_local.json `
    --mode execute `
    --output runs/live_smoke_tmax_2000/live_smoke_report.json `
    --output-root runs/live_smoke_tmax_2000 `
    --confirm-live I_UNDERSTAND_THIS_USES_CDS
```

This is intentionally constrained to:

```text
era5_daily_stats__tmax__2000
```

The output root must be a scratch directory such as
`runs/live_smoke_tmax_2000`, not `runs/dev_region`.

Audit the live smoke result:

```powershell
.\.venv\Scripts\python.exe scripts\08_audit_live_smoke.py `
    --config configs/rbmn_local.json `
    --mode audit `
    --live-report runs/live_smoke_tmax_2000/live_smoke_report.json `
    --output runs/live_smoke_tmax_2000/live_smoke_audit_report.json `
    --expected-output-root runs/live_smoke_tmax_2000
```

Inspect `artifact_hashes`, `product_validations`, and
`execution_status` before scaling up. Nothing under
`runs/live_smoke_tmax_2000/` should be committed.

## Outputs And Git Policy

Versioned reference artifacts live under:

```text
runs/dev_region/*.json
```

Raw and generated NetCDF outputs are ignored:

```text
runs/**/raw/
runs/**/intermediate/
runs/**/derived/
*.nc
*.nc4
runs/live_smoke*/
```

This keeps the repository light while preserving deterministic manifests
for review.

## Current Maturity

Ready:

- local dry-run pipeline
- manifest validation
- synthetic NetCDF preprocessing and index tests
- one-request live smoke readiness
- post-live audit tooling

Not ready:

- full live acquisition
- Docker execution
- Cloud Run / Google Cloud Storage execution
- legacy numeric regression
- production-scale operational monitoring

## Notes For Future Expansion

Future work should proceed in small slices:

1. Run and audit the one-request live smoke test.
2. If successful, plan a small multi-request live batch.
3. Add legacy numeric regression on governed sample NetCDFs.
4. Add Docker local execution.
5. Add Google Cloud Storage and Cloud Run Jobs.

CHIRPS, Daymet, Livneh, and WRF should remain paused unless the project
explicitly decides to expand beyond ERA5-Land.
