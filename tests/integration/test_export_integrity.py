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
