# CLI reference

Every subcommand has full help, for example `calibcheck report --help`. Precedence is:
command-line options, then `--config FILE.toml`, then built-in defaults. The resolved
configuration is written to `config.toml` with every report.

| Command | Purpose | Needs outcomes? |
|---|---|---|
| `calibcheck demo --out DIR` | Synthetic 90%-vs-65% walkthrough: report, fit, apply, held-out comparison | — |
| `calibcheck example [OUT.csv]` | Write the small bundled synthetic example | — |
| `calibcheck inspect FILE` | Validate and summarise only. No fitting, no resampling. `--json` for automation | yes |
| `calibcheck report FILE --prob P [--prob Q]` | Metrics, bins, figures, findings, JSON, CSV and HTML | yes |
| `calibcheck compare FILE --prob P --prob Q` | Paired comparison on identical rows (`--other FILE2` joins by ID) | yes |
| `calibcheck fit CAL.csv --prob P --method M --out art.json` | Fit a recalibrator on calibration data (in-sample) | yes |
| `calibcheck apply FILE --artifact art.json --prob P --out out.csv` | Add a recalibrated column | **no** |
| `calibcheck recalibrate --calibration CAL --test TEST --prob P` | Fit on calibration rows, evaluate raw and recalibrated on test rows | yes |
| `calibcheck app` | Local Streamlit interface, bound to 127.0.0.1 (`[app]` extra) | — |
| `calibcheck schema --out DIR` | Write JSON Schemas for reports, inspect output and artifacts | — |
| `calibcheck version` | Show the version and schema versions | — |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. Warnings may be shown, for example single-class data. |
| 1 | Invalid input, configuration or artifact (`INPUT_*`, `LABEL_*`, `PROB_*`, `CONFIG_INVALID`, `ARTIFACT_INVALID`) |
| 2 | Usage error (unknown option or missing argument) |
| 3 | Data-role protocol violation (`ROLE_OVERLAP`, `ROLE_ENTITY_OVERLAP`, `ROLE_TEMPORAL_ORDER`) |
| 4 | Unsafe output request: existing file or directory without `--overwrite`, a column collision, or writing outside the output directory |

## Output safety

* `--out DIR` must be new, empty, or hold a previous CalibCheck export. `--overwrite` replaces
  only the files listed in that export's `export_manifest.json`.
* A failed export leaves `_INCOMPLETE` and `"complete": false` in the manifest.
* `apply` refuses to write over its input file, an existing output file or an existing column
  unless you pass the corresponding flag explicitly.
* Machine-readable output: `--json` prints the full result to stdout. `--quiet` prints only
  the report path.
