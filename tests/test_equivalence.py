"""Optional extraction check against a separate original website checkout."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from dbmetrics import METRICS, measure_video
from dbmetrics._runtime import STAGE_FIELDS


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()
                if key not in {'run_provenance', 'timeout_seconds', 'worker_diagnostics'}}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


@unittest.skipUnless(os.environ.get('DBP_ORIGINAL_CPU_ROOT'),
                     'set DBP_ORIGINAL_CPU_ROOT to compare with the original checkout')
class EquivalenceTests(unittest.TestCase):
    def test_fixed_videos_match_original_records(self):
        original = Path(os.environ['DBP_ORIGINAL_CPU_ROOT']).resolve()
        ffmpeg = os.environ.get('DBP_TEST_FFMPEG', 'ffmpeg')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = Path(__file__).resolve().parents[1] / 'notebooks/data/nsd_subj01.mp4'
            shutil.copyfile(demo, root / 'demo_part_00000000.mp4')
            for name, colors in [('still', [32] * 20), ('cut', [0] * 10 + [255] * 10)]:
                path = root / (name + '_part_00000000.mp4')
                subprocess.run([ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                '-s', '16x16', '-r', '10', '-i', 'pipe:0', '-an',
                                '-c:v', 'libx264', '-threads', '1', '-crf', '0', str(path)],
                               input=b''.join(bytes([color]) * 16 * 16 * 3 for color in colors),
                               capture_output=True, check=True, timeout=10)
            real_ids = set()
            media_root = os.environ.get('DBP_EQUIVALENCE_MEDIA_ROOT')
            if media_root:
                candidates = sorted(Path(media_root).rglob('*.mp4'),
                                    key=lambda path: hashlib.sha256(path.name.encode()).digest())
                limit = int(os.environ.get('DBP_EQUIVALENCE_LIMIT', '50'))
                for candidate in candidates:
                    if candidate.stem in real_ids:
                        continue
                    shutil.copyfile(candidate, root / candidate.name)
                    real_ids.add(candidate.stem)
                    if len(real_ids) == limit:
                        break
                self.assertEqual(len(real_ids), limit, 'not enough unique real videos')
            completed_counts = dict.fromkeys(METRICS, 0)
            for path in sorted(root.glob('*.mp4')):
                actual = measure_video(path, ffmpeg=ffmpeg, timeout=60)
                expected = []
                for stage, fields in STAGE_FIELDS.items():
                    options = {'clip': {'clip_id': path.stem}, 'media_root': str(root),
                               'media_kind': 'original', 'ffprobe': 'ffprobe', 'ffmpeg': ffmpeg,
                               'timeout': 60, 'max_media_bytes': 2 * 1024 ** 3,
                               'cuts': False, 'cut_threshold': 27.0, 'stage': stage,
                               'ffmpeg_cut_threshold': 0.3}
                    result = subprocess.run([sys.executable, '-I', '-B',
                                             str(original / 'pilot_extract.py'), '--worker',
                                             json.dumps(options)], capture_output=True,
                                            check=True, timeout=65)
                    expected.extend(record for record in json.loads(result.stdout)['records']
                                    if record['field_id'] in fields)
                with self.subTest(video=path.stem):
                    self.assertEqual(normalized(actual), normalized(expected))
                    payload = {'schema_version': 'dbt1-pilot-measurements-v1',
                               'manifest_sha256': 'a' * 64, 'run_id': 'package-equivalence',
                               'records': actual}
                    validation = subprocess.run([
                        sys.executable, '-c',
                        'import json,sys; sys.path.insert(0,sys.argv[1]); '
                        'from pilot_build import validate_measurements; '
                        'payload=json.load(sys.stdin); '
                        'result=validate_measurements(payload,"b"*64,"a"*64,{sys.argv[2]:{}}); '
                        'print(len(result))', str(original), path.stem],
                        input=json.dumps(payload).encode(), capture_output=True,
                        check=True, timeout=15)
                    self.assertEqual(int(validation.stdout), 16)
                    by_field = {record['field_id']: record for record in actual}
                    self.assertEqual(by_field['duration_seconds']['processing_status'], 'complete')
                    if path.stem not in real_ids:
                        self.assertEqual(by_field['brightness_mean']['processing_status'], 'complete')
                        self.assertEqual(by_field['hard_cut_count']['processing_status'], 'complete')
                    for field, record in by_field.items():
                        if record['processing_status'] == 'complete':
                            completed_counts[field] += 1
                    if path.stem.startswith('still'):
                        self.assertEqual(by_field['hard_cut_count']['value'], 0)
                    if path.stem.startswith('cut'):
                        self.assertEqual(by_field['hard_cut_count']['value'], 1)
            print(json.dumps({'compared_videos': 3 + len(real_ids),
                              'real_videos': len(real_ids),
                              'completed_measurements': completed_counts}, sort_keys=True))
