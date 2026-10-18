"""CLI workflows, exit codes and API/CLI numerical agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from calibcheck import synthetic
from calibcheck.analysis import analyze_predictions
from calibcheck.cli import app

runner = CliRunner()


@pytest.fixture
def two_model_csv(tmp_path: Path) -> Path:
    return synthetic.generate("two_models", n=800, seed=4).write(tmp_path / "preds.csv")


def run(*args: str):
    return runner.invoke(app, [str(a) for a in args])


def test_version_and_help():
    r = run("version")
    assert r.exit_code == 0 and "calibcheck 0.1.0" in r.output
    r = run("--help")
    for cmd in ("demo", "inspect", "report", "compare", "fit", "apply", "app", "recalibrate"):
        assert cmd in r.output


def test_inspect_valid_and_invalid(tmp_path: Path, two_model_csv: Path):
    r = run("inspect", two_model_csv, "--prob", "p_model", "--id", "row_id", "--json")
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["valid"] and data["validation_only"] and data["dataset"]["n_rows_retained"] == 800
    bad = tmp_path / "bad.csv"
    bad.write_text("row_id,y_true,p_model\n1,1,1.7\n")
    r = run("inspect", bad, "--prob", "p_model", "--id", "row_id")
    assert r.exit_code == 1


def test_report_command_shape_from_spec(tmp_path: Path, two_model_csv: Path):
    out = tmp_path / "reports" / "example"
    r = run("report", two_model_csv, "--label", "y_true", "--prob", "p_model", "--id", "row_id",
            "--out", out)
    assert r.exit_code == 0, r.output
    for f in ("report.json", "report.html", "summary.txt", "config.toml", "tables/bins.csv",
              "figures/reliability.png", "export_manifest.json"):
        assert (out / f).exists(), f
    cfg_text = (out / "config.toml").read_text()
    assert "n_bins = 10" in cfg_text and "clip_eps = 1e-15" in cfg_text


def test_api_and_cli_agree(tmp_path: Path, two_model_csv: Path):
    out = tmp_path / "r"
    r = run("report", two_model_csv, "--prob", "p_model", "--id", "row_id", "--bootstrap",
            "--n-resamples", "300", "--seed", "11", "--out", out, "--no-figures", "--quiet")
    assert r.exit_code == 0, r.output
    cli = json.loads((out / "report.json").read_text())
    api = analyze_predictions(two_model_csv, columns={"label": "y_true", "probs": ["p_model"],
                                                      "id": "row_id"},
                              uncertainty={"bootstrap": True, "n_resamples": 300, "seed": 11})
    for key in ("brier", "log_loss", "ece", "roc_auc"):
        assert cli["models"][0]["metrics"][key]["value"] == api.models[0].metrics[key].value
    assert (cli["models"][0]["metrics"]["brier"]["interval"]["low"]
            == api.models[0].metrics["brier"].interval.low)


def test_config_file_and_cli_precedence(tmp_path: Path, two_model_csv: Path):
    conf = tmp_path / "c.toml"
    conf.write_text('[columns]\nlabel = "y_true"\nprobs = ["p_model"]\nid = "row_id"\n\n'
                    '[binning]\nn_bins = 20\nstrategy = "quantile"\n')
    r = run("report", two_model_csv, "--config", conf, "--bins", "5", "--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["config"]["binning"]["n_bins"] == 5
    assert data["config"]["binning"]["strategy"] == "quantile"
    conf.write_text('[binning]\nnbins = 20\n')
    r = run("report", two_model_csv, "--config", conf, "--prob", "p_model")
    assert r.exit_code == 1 and "nbins" in r.output


def test_compare_same_file_and_separate_files(tmp_path: Path, two_model_csv: Path):
    r = run("compare", two_model_csv, "--prob", "p_model", "--prob", "p_model_b", "--id",
            "row_id", "--bootstrap", "--n-resamples", "300", "--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    brier = next(c for c in data["comparisons"] if c["metric"] == "brier")
    assert brier["candidate"] == "p_model_b" and brier["n"] == 800
    import pandas as pd
    df = pd.read_csv(two_model_csv, dtype={"row_id": str})
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    df[["row_id", "y_true", "p_model"]].to_csv(a, index=False)
    df[["row_id", "y_true", "p_model_b"]].sample(frac=1, random_state=0).to_csv(b, index=False)
    r2 = run("compare", a, "--prob", "p_model", "--id", "row_id", "--other", b,
             "--other-prob", "p_model_b", "--json")
    assert r2.exit_code == 0, r2.output
    d2 = json.loads(r2.output)
    b2 = next(c for c in d2["comparisons"] if c["metric"] == "brier")
    assert b2["delta"] == pytest.approx(brier["delta"], rel=1e-12)


def test_fit_apply_and_protocol_exit_codes(tmp_path: Path):
    sc = synthetic.generate("overconfident", n=2000)
    cal = sc.frame[sc.frame.split == "calibration"].drop(columns="latent_p_true")
    test = sc.frame[sc.frame.split == "test"].drop(columns="latent_p_true")
    cal_p, test_p = tmp_path / "calibration.csv", tmp_path / "test.csv"
    cal.to_csv(cal_p, index=False)
    test.to_csv(test_p, index=False)
    art = tmp_path / "sig.json"
    r = run("fit", cal_p, "--prob", "p_model", "--id", "row_id", "--method", "sigmoid",
            "--out", art)
    assert r.exit_code == 0, r.output and "in-sample" in r.output
    assert run("fit", cal_p, "--prob", "p_model", "--out", art).exit_code == 4  # exists
    unlabeled = tmp_path / "unlabeled.csv"
    test.drop(columns="y_true").to_csv(unlabeled, index=False)
    out_csv = tmp_path / "applied.csv"
    r = run("apply", unlabeled, "--artifact", art, "--prob", "p_model", "--id", "row_id",
            "--out", out_csv)
    assert r.exit_code == 0, r.output
    import pandas as pd
    applied = pd.read_csv(out_csv, dtype={"row_id": str})
    assert "p_model__sigmoid" in applied and applied["row_id"].iloc[0].startswith("00")
    assert run("apply", unlabeled, "--artifact", art, "--prob", "p_model", "--out",
               out_csv).exit_code == 4
    assert run("apply", unlabeled, "--artifact", art, "--prob", "other", "--out",
               tmp_path / "x.csv").exit_code == 1  # association mismatch
    # evaluating on the calibration file itself is refused (exit 3)
    r = run("report", cal_p, "--prob", "p_model", "--id", "row_id", "--fitted-artifact", art)
    assert r.exit_code == 3
    r = run("report", test_p, "--prob", "p_model", "--id", "row_id", "--fitted-artifact", art,
            "--json")
    assert r.exit_code == 0


def test_recalibrate_and_overlap_refusal(tmp_path: Path):
    sc = synthetic.generate("overconfident", n=2000)
    df = sc.frame.drop(columns="latent_p_true")
    cal_p, test_p = tmp_path / "cal.csv", tmp_path / "test.csv"
    df[df.split == "calibration"].to_csv(cal_p, index=False)
    df[df.split == "test"].to_csv(test_p, index=False)
    out = tmp_path / "recal"
    r = run("recalibrate", "--calibration", cal_p, "--test", test_p, "--prob", "p_model",
            "--id", "row_id", "--out", out, "--no-figures", "--quiet")
    assert r.exit_code == 0, r.output
    assert (out / "artifacts" / "sigmoid.json").exists()
    data = json.loads((out / "report.json").read_text())
    assert data["kind"] == "recalibration" and data["protocol"]["mode"] == "exploratory"
    r = run("recalibrate", "--calibration", cal_p, "--test", cal_p, "--prob", "p_model",
            "--id", "row_id")
    assert r.exit_code == 3


def test_demo_runs(tmp_path: Path):
    r = run("demo", "--out", tmp_path / "demo", "--n", "1200", "--n-resamples", "200")
    assert r.exit_code == 0, r.output
    assert (tmp_path / "demo" / "reports" / "raw" / "report.html").exists()
    assert (tmp_path / "demo" / "reports" / "recalibration" / "report.html").exists()
    assert "SYNTHETIC" in r.output


def test_schema_command(tmp_path: Path):
    r = run("schema", "--out", tmp_path / "s")
    assert r.exit_code == 0
    s = json.loads((tmp_path / "s" / "report.schema.json").read_text())
    assert s["title"] == "AnalysisResult"


def test_exported_config_reproduces_report_for_dotted_column(tmp_path: Path):
    """config.toml from a report reloads unchanged and reproduces the same results."""
    import pandas as pd

    from calibcheck.config import load_config_file, resolve_config

    sc = synthetic.generate("overconfident", n=600, seed=9)
    df = pd.DataFrame({"row_id": sc.frame.row_id, "y_true": sc.frame.y_true,
                       "model.v1": (sc.frame.p_model * 100).round(4)})  # percent scale
    src = tmp_path / "input data.csv"
    df.to_csv(src, index=False)
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    r1 = run("report", src, "--prob", "model.v1", "--transform", "model.v1=percent", "--id",
             "row_id", "--bootstrap", "--n-resamples", "200", "--out", out1, "--quiet")
    assert r1.exit_code == 0, r1.output
    text = (out1 / "config.toml").read_text(encoding="utf-8")
    assert '"model.v1" = "percent"' in text
    first = json.loads((out1 / "report.json").read_text())
    reloaded = resolve_config(load_config_file(out1 / "config.toml"))
    assert reloaded.model_dump(mode="json") == first["config"]
    # Reproduce from the exported file alone, with no manual repair and no column flags.
    r2 = run("report", src, "--config", out1 / "config.toml", "--out", out2, "--quiet")
    assert r2.exit_code == 0, r2.output
    second = json.loads((out2 / "report.json").read_text())
    assert first["manifest"]["results_digest"] == second["manifest"]["results_digest"]
    first.pop("manifest")
    second.pop("manifest")
    assert first == second
    assert any(f["code"] == "TRANSFORM_APPLIED" for f in second["validation"])
