# Data Sources

## Canonical Case Study Polygon

- Name: `rbmn`
- Location: `01_data/case_studies/rbmn.geojson`
- Format: GeoJSON `FeatureCollection`
- Geometry type: single-feature `MultiPolygon`
- CRS: `urn:ogc:def:crs:OGC:1.3:CRS84` in file metadata; treat as longitude/latitude and normalize to `EPSG:4326` in pipeline manifests.
- Feature name: `Marismas Nacionales Nayarit`
- State: `Nayarit`
- Bounding box, west/south/east/north: `[-105.70053541497737, 21.741759030754736, -105.2937762317876, 22.573077683788142]`
- Bounding box, Copernicus area order north/west/south/east: `[22.573077683788142, -105.70053541497737, 21.741759030754736, -105.2937762317876]`
- Use: primary polygon for all local tests, fixtures, dry-run download manifests, and pipeline examples until further notice.
- Validation entry point: `scripts/00_validate_region.py` derives the bbox above from this file at runtime and writes a deterministic `region_manifest.json` for the rest of the pipeline. The numbers above are what that script must produce; any drift is a regression.

## ERA5-Land Sources From Legacy Workflow

### Daily Statistics

- Dataset: `derived-era5-land-daily-statistics`
- Variables: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`
- Derived project variables: `tmax`, `tmin`, `tmean`, `u10m`, `v10m`
- Access method: Copernicus CDS API
- First implementation mode: dry-run request manifest only; live download later.
- Daily statistics mapping (from `lib/download_plan.py`):
  - `tmax` <- `2m_temperature` `daily_maximum`
  - `tmin` <- `2m_temperature` `daily_minimum`
  - `tmean` <- `2m_temperature` `daily_mean`
  - `u10m` <- `10m_u_component_of_wind` `daily_mean`
  - `v10m` <- `10m_v_component_of_wind` `daily_mean`
- Planned chunking: one request per (project variable, year). For 2000-2024
  this is 5 * 25 = 125 daily-statistics requests. Legacy `time_zone` =
  `utc-07:00` and `frequency` = `6_hourly` are preserved as planned values
  (their scientific correctness is an open decision, not a finding).

### Hourly Precipitation

- Dataset: `reanalysis-era5-land`
- Variable: `total_precipitation`
- Derived project variable: `pr`
- Access method: Copernicus CDS API
- First implementation mode: dry-run request manifest only; live download later.
- Planned chunking: two semester requests per year (`H1` = months 01-06,
  `H2` = months 07-12). For 2000-2024 this is 25 * 2 = 50 precipitation
  requests. `data_format` = `netcdf`, `download_format` = `unarchived`.
- Precipitation timezone policy is intentionally left as an open decision
  (see `90_legacy_review/migration_decision_log.md`); the planned manifest
  records it as `time_zone_policy: open` rather than baking in `utc-07:00`.

## CDS Payload Validation Evidence Status (Milestone 003)

The planned daily-statistics payload keys M002 generates
(`variable`, `daily_statistic`, `year`, `month`, `day`, `time_zone`,
`frequency`, `area`) and the planned hourly-precipitation payload keys
(`variable`, `year`, `month`, `day`, `time`, `data_format`,
`download_format`, `area`) were chosen from the legacy
`Download_tmax_tmin_tmean_wind.py` / hourly precipitation scripts and the
CDS dataset form names. They have **not yet** been validated against a live
CDS submission from this repository because doing so requires an
owner-authorized account and is out of scope for automated tests.

Until that live validation happens:

- `scripts/02_download_era5_land.py` defaults to `--mode dry-run` and only
  the dry-run path is automatically test-covered.
- `--mode execute` exists and calls `cdsapi.Client.retrieve(dataset,
  payload, target)` with the planned payload as-is. The first live run
  should be a manual, owner-authorized acquisition of a single small
  request (e.g. one `tmax` year) followed by a payload audit before scaling
  up.
- Daily-statistic string normalization: legacy used `daily_max` /
  `daily_min` but `lib/download_plan.py` records `daily_maximum` /
  `daily_minimum` to match current CDS form labels. If the live test
  reveals the endpoint accepts the legacy strings only, this is the place
  to record the correction.

## Daily Standard Product Schema (Milestone 004)

`scripts/03_preprocess_daily.py` writes one daily NetCDF per
(`project_variable`, `year`) under
`runs/{run_id}/intermediate/daily/{project_variable}/{year}.nc`.

Coordinate normalization (applied in execute mode):

- `valid_time` (CDS naming) renamed to `time`.
- `latitude` / `longitude` renamed to `lat` / `lon`.

Variable normalization:

- The source CDS variable name (e.g. `t2m`, `u10`, `v10`) is renamed to
  the project variable (`tmax`, `tmin`, `tmean`, `u10m`, `v10m`).

Unit conversion:

- `tmax`, `tmin`, `tmean`: Kelvin -> Celsius (`value - 273.15`),
  `units = "degC"`, `preprocessing_offset_kelvin = 273.15` recorded as
  attributes.
- `u10m`, `v10m`: no unit conversion.

Spatial clipping:

- Each `(lat, lon)` cell whose center lies outside the region polygon
  (from `region_manifest.geometry_path`) is set to NaN via
  `shapely.contains_xy` against the polygon union.

Out of scope for M004 (closed):

- Climate indices (M005, see below).

As of M006, hourly precipitation requests are no longer always deferred:
when `scripts/03_preprocess_daily.py` is invoked with
`--precipitation-policy legacy_utc_minus_7` the precipitation pipeline
runs (see "Precipitation Daily Standard Product Schema (Milestone 006)"
below). When the flag is omitted, M004 behavior is preserved: hourly
precipitation requests are recorded as `status = deferred`,
`reason = precipitation_policy_open`.

## Precipitation Daily Standard Product Schema (Milestone 006)

`scripts/03_preprocess_daily.py --precipitation-policy
legacy_utc_minus_7` writes one daily NetCDF per year under
`runs/{run_id}/intermediate/daily/pr/{year}.nc`. H1 (months 01-06) and
H2 (months 07-12) acquisition chunks for the same year are merged into
a single daily product.

Coordinate normalization (applied in execute mode):

- `valid_time` (CDS naming) renamed to `time`.
- `latitude` / `longitude` renamed to `lat` / `lon`.

Variable normalization:

- The source CDS variable (`tp` or `total_precipitation`) is renamed
  to `pr`.

Temporal policy `legacy_utc_minus_7`:

- Time coordinate is shifted by `-7h` (fixed offset, no DST) before
  daily aggregation. This matches the documented intent of the
  legacy `Etc/GMT+7` workflow without reproducing its
  `time`/`valid_time` coordinate-mixing bug or its double-diff dance.

Deaccumulation:

- ERA5-Land `tp` is hourly accumulated precipitation in meters.
  `lib.precipitation.deaccumulate_hourly_tp` takes a `.diff('time')`
  along the time dimension and clamps negatives to zero. This handles
  the daily reset (where the accumulation drops back to zero) and
  tiny negative artifacts in the same step. The first hour of each
  series is lost to the diff -- documented as a known limitation in
  `90_legacy_review/legacy_risks.md`.

Unit conversion:

- Hourly increments (meters) are summed per local-clock day and
  multiplied by `1000` to obtain `mm/day`. The output NetCDF records
  `units = "mm/day"`, `source_units = "m"`,
  `preprocessing_conversion_factor = 1000.0`, and the active
  `precipitation_policy` as data-array attributes so an auditor can
  reconstruct the math from the file alone.

Spatial clipping:

- The same polygon mask M004 applies to daily-statistics is reused
  via `lib.preprocessing.apply_region_mask`.

Open scientific question (not closed by M006):

- Whether future production should keep the fixed `legacy_utc_minus_7`
  offset, use UTC, or use a region-specific civil-time policy
  (especially relevant for Nayarit, which observes DST in some years).
  See `90_legacy_review/migration_decision_log.md` "Open Decisions".

Out of scope for M006:

- Precipitation indices (`PRCPTOT`, `RX1day`, `R95p`, `CDD`, `CWD`,
  `R10mm`, `R20mm`).
- A region-specific or DST-aware timezone policy.
- Backfilling the first lost hour of each accumulation series.

## Annual Temperature Index Schema (Milestone 005)

`scripts/04_compute_indices.py` writes one annual NetCDF per index
under `runs/{run_id}/derived/indices/{index_id}.nc`. Each NetCDF
contains a single variable named after the index with a `time`
dimension carrying one timestamp per year (annual-end frequency,
xarray `resample(time="YE")`).

Index families implemented:

| Index | Required variable(s) | Formula | Units |
|---|---|---|---|
| `Tmx` | `tmean` | annual max of daily mean | degC |
| `Tmn` | `tmean` | annual min of daily mean | degC |
| `TXx` | `tmax` | annual max of daily max | degC |
| `TNn` | `tmin` | annual min of daily min | degC |
| `DTR` | `tmax`, `tmin` | annual mean of `(tmax - tmin)` | degC |
| `SU` | `tmax` | annual count of days with `tmax > 30 degC` | days |
| `TR` | `tmin` | annual count of days with `tmin > 20 degC` | days |

Each output NetCDF records `units`, `description`, and (for `SU` /
`TR`) `threshold_degc` as attributes.

Out of scope for M005:

- Wind indices (`RAP_med`, `RAP_mod`, `RAP90p`, `Dir`, `Dir90p`,
  `DVIn90p`, `DVIb10p`).
- Percentile indices (`TX90p`, `TN10p`).
- Spell-duration indices (`WSDI`).

Precipitation indices are now implemented in M007 (see below).

## Annual Precipitation Index Schema (Milestone 007)

`scripts/04_compute_indices.py --index-family precipitation` writes
one annual NetCDF per index under
`runs/{run_id}/derived/indices/{index_id}.nc` (same layout as M005
temperature indices). Each NetCDF carries a single variable named
after the index on a `time` dimension with one timestamp per year
(annual-end frequency, xarray `resample(time="YE")`).

Indices implemented:

| Index    | Required | Formula                                          | Units |
|----------|----------|--------------------------------------------------|-------|
| `PRCPTOT`| `pr`     | annual sum of wet-day precipitation (`pr >= 1 mm`) | mm  |
| `RX1day` | `pr`     | annual maximum daily precipitation               | mm    |
| `R95p`   | `pr`     | annual sum of `pr` on days where `pr > RRwn95`   | mm    |
| `CDD`    | `pr`     | annual max consecutive run of dry days (`pr < 1 mm`) | days |
| `CWD`    | `pr`     | annual max consecutive run of wet days (`pr >= 1 mm`) | days |
| `R10mm`  | `pr`     | annual count of days with `pr >= 10 mm`          | days  |
| `R20mm`  | `pr`     | annual count of days with `pr >= 20 mm`          | days  |

Each output NetCDF records `units`, `description`, and the
index-specific thresholds (`wet_day_threshold_mm`, `threshold_mm`)
as attributes. `R95p` additionally records `percentile = 0.95`
and `baseline_period_policy = "period_full_extent"` so the
reference period for `RRwn95` is auditable from the file alone.

The CLI flag `--index-family` (default `temperature`) selects which
family runs; the M005 reference command is byte-identical because
`temperature` is the default.

Out of scope for M007:

- Wind indices.
- Temperature percentile indices (`TX90p`, `TN10p`).
- Spell-duration indices (`WSDI`).
- DST-aware or region-specific civil-time precipitation policy. The
  `pr` daily products are produced by M006 under the compatibility
  policy `legacy_utc_minus_7`; the final temporal-policy decision
  remains open in `90_legacy_review/migration_decision_log.md`.

## Live Smoke Readiness Allowlist (Milestone 010)

The first reviewed live ERA5-Land acquisition from this repository is
intentionally bounded to one daily-statistics request and runs only
through `scripts/07_run_live_smoke.py` with explicit owner
authorization. The allowlist is part of the M010 contract:

- Allowed request id: `era5_daily_stats__tmax__2000` (the
  `2m_temperature` / `daily_maximum` request for year 2000 from the
  canonical M002 download plan).
- Allowed indices for the smoke run: `TXx`, `SU` (the two simple
  annual indices computable from `tmax` alone). All other temperature
  indices and every precipitation index are explicitly out of scope
  for the smoke.
- Default scratch output root: `runs/live_smoke_tmax_2000/`. Execute
  mode rejects `--output-root runs/dev_region` so the canonical
  reference manifests stay byte-frozen.
- Execute prerequisites: `cdsapi` installed plus CDS credentials in
  `~/.cdsapirc` or env vars `CDSAPI_URL` / `CDSAPI_KEY`, plus the
  exact confirmation token `I_UNDERSTAND_THIS_USES_CDS` passed via
  `--confirm-live`.
- Preflight mode is the safe default: no network, no credentials,
  no NetCDF writes. The canonical preflight artifact lives at
  `runs/dev_region/live_smoke_plan.json` and records the M009
  validator outcome plus SHA-256 hashes of the four prerequisite
  manifests (region / download / pipeline / validation_report).
- Out of scope for M010: full 175-request live execution,
  precipitation live execution, wind / percentile / spell-duration
  indices, Docker, Cloud Run, GCS, any change to `scripts/00`-`06`
  behavior, any change to the canonical M001-M009 reference
  manifests.
