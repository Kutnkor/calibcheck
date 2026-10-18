"""Minimal downstream consumer of a CalibCheck report.json.

Checks the schema version before reading, and treats ``null`` values as unavailable
(the reason code says why) instead of assuming a number is always present.

Usage:  python examples/consume_report.py path/to/report.json
"""

from __future__ import annotations

import json
import sys

SUPPORTED_MAJOR = "1"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    version = str(report.get("schema_version", ""))
    if version.split(".")[0] != SUPPORTED_MAJOR:
        print(f"unsupported report schema {version!r}", file=sys.stderr)
        return 2
    for model in report["models"]:
        for key in ("brier", "log_loss", "ece"):
            m = model["metrics"][key]
            if m["value"] is None:
                print(f"{model['name']}\t{key}\tunavailable ({m['reason']})")
                continue
            iv = m.get("interval") or {}
            ci = f"\t[{iv['low']:.4f}, {iv['high']:.4f}]" if iv.get("status") == "ok" else ""
            extra = f"\t{m['details'].get('partition', '')}" if key == "ece" else ""
            print(f"{model['name']}\t{key}\t{m['value']:.4f}{ci}{extra}")
    for c in report.get("comparisons", []):
        print(f"compare\t{c['metric']}\t{c['candidate']} - {c['reference']}\t{c['delta']}\t{c['favours']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
