"""Output safety and numeric precision regression tests."""
import json

import pytest

from calibcheck.errors import OutputError
from calibcheck.report import MANIFEST_NAME, prepare_output_dir


@pytest.mark.parametrize("manifest", [[], {}, {"tool": "other", "files": {}},
                                      {"tool": "calibcheck", "files": []}])
def test_foreign_or_malformed_manifest_never_authorizes_deletion(tmp_path, manifest):
    note = tmp_path / "notes.txt"
    note.write_text("preserve")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(OutputError):
        prepare_output_dir(tmp_path, overwrite=True)
    assert note.read_text() == "preserve"


def test_manifest_is_fully_checked_before_deleting_any_file(tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("preserve")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({
        "tool": "calibcheck", "files": {"notes.txt": "a" * 64, "../outside.txt": "b" * 64}}))
    with pytest.raises(OutputError):
        prepare_output_dir(tmp_path, overwrite=True)
    assert note.read_text() == "preserve" and (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.parametrize("text", [" =1+1", "  +SUM(A1)", "\n@formula", "\tplain", " -1+1", "\r=cmd"])
def test_csv_formulas_cannot_hide_behind_whitespace(text):
    import pandas as pd

    from calibcheck.report import csv_safe
    frame = pd.DataFrame({text: [text], "numeric": [-0.25]})
    safe = csv_safe(frame)
    assert safe.columns[0] == "'" + text
    assert safe.iloc[0, 0] == "'" + text
    assert safe.numeric.iloc[0] == -0.25
    assert frame.columns[0] == text  # caller data is unchanged


def test_apply_output_preserves_full_float64_precision(tmp_path):
    import csv

    from typer.testing import CliRunner

    from calibcheck.calibration import fit_calibrator, save_artifact
    from calibcheck.cli import app
    from calibcheck.config import AnalysisConfig
    from calibcheck.data import load_probabilities_only
    src, dest, art_path = tmp_path / "in.csv", tmp_path / "out.csv", tmp_path / "map.json"
    src.write_text("p\n0.12345678901234567\n0.9999999999999999\n")
    _, expected = load_probabilities_only(src, "p", AnalysisConfig())
    save_artifact(fit_calibrator([0.2, 0.8], [0, 1], "identity", model_association="p"), art_path)
    result = CliRunner().invoke(app, ["apply", str(src), "--artifact", str(art_path),
                                       "--prob", "p", "--out", str(dest)])
    assert result.exit_code == 0, result.output
    with dest.open() as fh:
        actual = [float(row["p__identity"]) for row in csv.DictReader(fh)]
    assert actual == expected.tolist()
