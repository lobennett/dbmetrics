"""Isolated entry point; each stage has its own killable process group."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dbp_metrics._runtime import extract_clip, strict_json

if __name__ == '__main__':
    options = strict_json(sys.argv[1])
    timeout = options.pop('timeout')
    records, failed = extract_clip(deadline=time.monotonic() + timeout, **options)
    print(json.dumps({'records': records, 'failed': failed}, allow_nan=False))
