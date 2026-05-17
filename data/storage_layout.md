# Storage Layout

## Canonical Test Geometry

```text
01_data/
  case_studies/
    rbmn.geojson
```

`01_data/case_studies/rbmn.geojson` is the canonical polygon for development and tests until further notice. Do not copy this geometry into generated run folders by hand; pipeline scripts should reference it as an input and write normalized manifests under `runs/`.

## Proposed Runtime Layout

```text
runs/
  {run_id}/
    region_manifest.json
    download_manifest.json
    preprocessing_manifest.json
    index_manifest.json
    validation_report.md
    raw/
    intermediate/
      daily/
    derived/
      indices/
```

The same path structure should work locally, inside Docker, and later when mapped to Google Cloud Storage.

## Implemented As Of Milestone 005

`region_manifest.json` is produced by `scripts/00_validate_region.py` (M001).
`download_manifest.json` is produced by `scripts/01_plan_downloads.py` (M002)
and is a planned-only artifact: no Copernicus call, no NetCDF, no network.
`acquisition_manifest.json` is produced by
`scripts/02_download_era5_land.py` (M003) and records the dry-run plan or a
live execution result. The dry-run default writes only the manifest and
does not create any `.nc` files.
`preprocessing_manifest.json` is produced by
`scripts/03_preprocess_daily.py` (M004/M006) and records the dry-run
plan or a local NetCDF preprocessing run. M004 owns daily-statistics
variables (`tmax`, `tmin`, `tmean`, `u10m`, `v10m`). Hourly
precipitation requests are handled by policy: without
`--precipitation-policy`, each chunk is recorded as
`status = deferred` with reason `precipitation_policy_open` (M004
default); with `--precipitation-policy legacy_utc_minus_7`, M006
collapses H1 + H2 chunks for the same year into one planning unit
and writes a single daily `pr` product per year. The canonical
`runs/dev_region/preprocessing_manifest.json` is produced **without**
the flag, so its precipitation requests remain deferred.
`index_manifest.json` is produced by `scripts/04_compute_indices.py`
(M005) and records the dry-run plan or a local index-computation run
for simple annual temperature indices (`Tmx`, `Tmn`, `TXx`, `TNn`,
`DTR`, `SU`, `TR`). Precipitation, wind, percentile (`TX90p`,
`TN10p`), and spell-duration (`WSDI`) indices are deferred.

```text
runs/
  dev_region/
    region_manifest.json
    download_manifest.json
    acquisition_manifest.json
    preprocessing_manifest.json
    index_manifest.json
```

In execute mode, M004 writes daily standard products under
`runs/{run_id}/intermediate/daily/{project_variable}/{year}.nc`, M005
writes annual index products under
`runs/{run_id}/derived/indices/{index_id}.nc`, and M006 writes daily
precipitation (`pr`) products under
`runs/{run_id}/intermediate/daily/pr/{year}.nc` (the same M004 layout
extended). All three subtrees are ignored by `.gitignore`
(`runs/**/intermediate/`, `runs/**/derived/`, `*.nc`, `*.nc4`) so only
the manifests are versioned. The remaining manifest name listed above
(`validation_report.md`) is reserved for a later milestone.

Precipitation-specific reference manifests live alongside the
M001-M005 canonical artifacts so the older artifacts stay byte-frozen:

```text
runs/
  dev_region/
    acquisition_manifest_precipitation_dry_run.json   # M003 dry-run for the two 2000 pr chunks
    preprocessing_manifest_precipitation.json         # M006 dry-run for pr/2000.nc
    index_manifest_precipitation.json                 # M007 dry-run for the 7 precipitation indices
    pipeline_manifest.json                            # M008 orchestration summary (8 steps)
```

`pipeline_manifest.json` is the M008 orchestration summary written by
`scripts/05_run_pipeline.py`. It records each of the eight canonical
dry-run steps with their command argv, exit code, output path, and
SHA-256 hash so a reviewer can reproduce the entire local dry-run
sequence from a single artifact.

`validation_report.json` is the M009 validation/regression summary
written by `scripts/06_validate_outputs.py`. It reads the M008
pipeline manifest plus the M001-M007 manifests it references and
records the outcome of 17 stable-order checks (graph consistency,
hash recomputation, dry-run side-effect policy, daily / index
product schema placeholders). The canonical dry-run currently
records 15 passed + 2 skipped (the product-schema aggregator
checks are skipped because no NetCDF products exist on disk in
dry-run state).

`live_smoke_audit_plan.json` is the M011 audit-readiness artifact
written by `scripts/08_audit_live_smoke.py` in default `--mode
preflight`. It is network- and credential-free: it hashes the M010
preflight plan plus the M010 safety-corrections review, lists the
eleven audit checks the future audit run will execute in stable
order, and lists the six expected scratch-root product paths
(`raw_target_path`, `daily_tmax_path`, `TXx_index_path`,
`SU_index_path`, `live_report_path`, `audit_report_path`) the M010
execute run is expected to produce. The canonical preflight
artifact carries `manifest_type = era5_land_live_smoke_audit`,
`mode = preflight`, `requires_network = false`, and
`execution_status = ready_for_live_smoke_audit`. `--mode audit`
reads an M010 execute report under the scratch root, hashes the four
expected NetCDF products, runs the M009 per-file NetCDF validators
on the daily and index files, and writes
`runs/live_smoke_tmax_2000/live_smoke_audit_report.json` (covered by
the existing `runs/live_smoke*/` gitignore rule).

`live_smoke_plan.json` is the M010 owner-authorized live smoke-test
plan written by `scripts/07_run_live_smoke.py`. In default
`--mode preflight` it is network- and credential-free: it re-runs
the M009 validator dry-run, re-hashes the four prerequisite
manifests (region / download / pipeline / validation_report),
confirms the smoke `request_id` is in the M002 download manifest
and in the M010 allowlist (`era5_daily_stats__tmax__2000`), and
confirms the smoke `--output-root` is not the canonical
`runs/dev_region`. The canonical preflight artifact carries
`manifest_type = era5_land_live_smoke_plan`, `mode = preflight`,
`requires_network = false`, `execution_status =
ready_for_owner_authorized_live_test`, six passing preflight
checks, and four planned execute steps. `--mode execute` is
gated by the confirmation token `I_UNDERSTAND_THIS_USES_CDS` plus
a scratch `--output-root`; it writes its plan/report JSON under
the scratch root (default `runs/live_smoke_tmax_2000/`) which is
explicitly excluded from version control by `runs/live_smoke*/`
in `.gitignore`. The four execute outputs (`raw/era5_land/...`,
`intermediate/daily/tmax/2000.nc`, `derived/indices/TXx.nc`,
`derived/indices/SU.nc`) are also covered by the existing
`runs/**/raw/`, `runs/**/intermediate/`, `runs/**/derived/`, and
`*.nc` ignore rules.

## `runs/` Git Policy (Adopted In Milestone 003)

- Small canonical manifests under `runs/<run_id>/*.json` are versioned in
  the repo as reference outputs the reviewer can diff.
- Raw downloads, intermediate, and derived NetCDF/data products are
  **not** versioned. `.gitignore` ignores `runs/**/raw/`,
  `runs/**/intermediate/`, `runs/**/derived/`, and any `*.nc` / `*.nc4`
  file. Download requests planned in `download_manifest.json` reference
  output paths under `raw/era5_land/daily_statistics/{project_var}/{year}.nc`
  and `raw/era5_land/hourly_precipitation/{year}_{H1|H2}.nc`; those files
  only materialize when `scripts/02_download_era5_land.py --mode execute`
  runs against a live CDS account.

The scripts accept arbitrary `--output` / `--output-root` paths, so the
same code will work against mounted GCS paths in later milestones without
modification.
