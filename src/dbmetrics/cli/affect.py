#!/usr/bin/env python3
"""Offline CSV review validation, descriptive summaries, and affect evaluation."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys


from dbmetrics.experimental import affect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="write a CSV header to stdout")
    template.add_argument("--kind", choices=affect.FIELDS, default="reviews")
    validate = commands.add_parser("validate", help="validate the whole CSV; no persistence")
    validate.add_argument("csv", type=Path)
    validate.add_argument("--kind", choices=affect.FIELDS, default="reviews")
    importer = commands.add_parser("import-reviews", help="emit validated normalized review JSON")
    importer.add_argument("csv", type=Path)
    summary = commands.add_parser("summarize", help="summarize human reviews by video and construct")
    summary.add_argument("csv", type=Path)
    evaluation = commands.add_parser("evaluate", help="compare affect predictions with declared heldout reviews")
    evaluation.add_argument("csv", type=Path, help="human reviews")
    evaluation.add_argument("predictions", type=Path)
    prompts = commands.add_parser("export-prompts", help="export depicted-affect prompts; no inference")
    prompts.add_argument("--video-id", action="append", required=True)
    arguments = parser.parse_args(argv)
    inputs = []

    def load(path, kind):
        data = path.read_bytes()
        rows = affect.read_csv(io.StringIO(data.decode("utf-8-sig"), newline=""), kind)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "kind": kind})
        return rows

    try:
        if arguments.command == "template":
            csv.writer(sys.stdout).writerow(affect.FIELDS[arguments.kind])
            return 0
        if arguments.command == "export-prompts":
            result = affect.export_prompts(arguments.video_id)
        else:
            rows = load(arguments.csv, getattr(arguments, "kind", "reviews"))
            if arguments.command == "validate":
                result = {"valid": True, "n_rows": len(rows)}
            elif arguments.command == "import-reviews":
                result = {"reviews": rows, "n_rows": len(rows)}
            elif arguments.command == "summarize":
                result = {"summaries": affect.summarize(rows)}
            else:
                result = affect.evaluate(rows, load(arguments.predictions, "predictions"))
        result.update(schema_version=affect.SCHEMA_VERSION, inputs=inputs)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
