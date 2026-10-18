# Configuration (TOML)

Pass a file with `--config calibcheck.toml`. Unknown keys are errors, so a misspelling cannot
silently fall back to a default. The file is parsed as data and nothing in it is executed. The
values below are the defaults; `examples/calibcheck.toml` is a working example.

```toml
positive_label = "1"          # required unless labels are 0/1
missing_policy = "fail"       # or "drop" (drops and records invalid rows)
clip_eps = 1e-15              # log loss / logit clipping
threshold = 0.5               # threshold-accuracy view only
dependence = "independent"    # independent | entity_clustered | time_dependent | unknown
# reference_prevalence = 0.12 # training/development prevalence for the reference predictor
# time_window = "M"           # D | W | M | Q | Y
# reference_model = "p_a"     # comparisons against this column
pairwise = false              # all pairwise comparisons (explicit request)
max_file_mb = 500.0
max_columns = 2000

[transforms]                  # explicit, recorded; never implicit
# p_score = "percent"         # divide by 100
# p_logit = "logit"           # apply the logistic function

[columns]
label = "y_true"
probs = ["p_model_a"]         # never guessed
id = "row_id"
# entity = "customer_id"
# timestamp = "ts"
groups = []

[binning]
strategy = "uniform"          # or "quantile"
n_bins = 10
min_support = 30              # display heuristic
sensitivity = true
sensitivity_bins = [5, 10, 20]
sensitivity_strategies = ["uniform", "quantile"]

[uncertainty]
level = 0.95
wilson = true
bootstrap = false
n_resamples = 2000
seed = 20251018
bootstrap_bin_metrics = false # exploratory ECE intervals
max_elements_per_batch = 4000000

[groups]
max_groups = 12
min_group_size = 30
max_cardinality = 50

[privacy]
mode = "private"              # "public" omits paths, fingerprints, IDs, group names
min_cell = 10
reveal_group_names = false
export_retained_rows = false
```

The `config.toml` exported with every report reloads to an identical configuration. Column
names that are not plain identifiers are written as quoted keys, for example
`"model.v1" = "percent"` under `[transforms]`. Write them the same way in your own files, or
TOML will read `model.v1` as nested tables.

The Streamlit app builds the same `AnalysisConfig` object. Its Export section shows the
equivalent CLI command, and `config.toml` can be downloaded.

Clipping epsilon must be at least float64 machine epsilon (`2**-52`, approximately
`2.22e-16`) and below `1e-3`. Smaller values can round `1 - eps` to exactly one and
produce infinite log losses. The default remains `1e-15`.

Transformation keys must match selected probability columns. Unused or misspelled keys
are rejected during input validation rather than silently ignored. CSV columns must have
nonempty headers and each record must match the header width.
