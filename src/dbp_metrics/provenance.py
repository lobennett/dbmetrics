"""Bind runs to the complete installed computation package."""
import hashlib
import json
from pathlib import Path
import subprocess


def code_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}


def verify_code(expected):
    if expected != code_hashes():
        raise ValueError('dbp-metrics changed; start a new run with a new output directory')


def verify_interpreter(python, expected):
    command = [str(python), '-I', '-B', '-c',
               'import json; from dbp_metrics.provenance import code_hashes; '
               'print(json.dumps(code_hashes()))']
    result = subprocess.run(command, capture_output=True, timeout=30, check=True)
    if len(result.stdout) > 256 * 1024 or json.loads(result.stdout) != expected:
        raise ValueError('Worker interpreter must use the same pinned dbp-metrics package')
