import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from dbmetrics.experimental.scene_continuity import Config, analyze_frames, measure, run_bounded


class DiagnosticTests(unittest.TestCase):
    def analyze(self, levels):
        frames = np.broadcast_to(np.array(levels, dtype=np.uint8)[:, None, None],
                                 (len(levels), 4, 4)).copy()
        return analyze_frames(frames, np.arange(len(levels)) / 10, Config())

    def test_isolated_step_has_one_candidate_at_new_frame(self):
        result = self.analyze([16] * 10 + [235] * 10)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([row['timestamp_seconds'] for row in result['candidates']], [1.0])
        self.assertAlmostEqual(result['trace'][9]['change'], 219 / 255)
        self.assertEqual(result['sustained_change_runs'], [])

    def test_sustained_change_is_not_a_list_of_cuts(self):
        result = self.analyze([0, 64, 128, 192] * 6)
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['sustained_change_runs'][0]['transition_count'], 23)

    def test_flash_is_explicitly_ambiguous(self):
        result = self.analyze([16] * 10 + [235] + [16] * 10)
        self.assertEqual(len(result['candidates']), 2)
        self.assertTrue(all(row['flicker_ambiguous'] for row in result['candidates']))
        self.assertEqual(len(result['flicker_returns']), 1)

    def test_alternating_flicker_is_sustained_and_ambiguous(self):
        result = self.analyze([16, 235] * 10)
        self.assertEqual(result['candidates'], [])
        self.assertTrue(result['sustained_change_runs'])
        self.assertTrue(result['flicker_returns'])

    def test_static_and_short_input_are_not_confident_cuts(self):
        self.assertEqual(self.analyze([16] * 20)['candidates'], [])
        result = self.analyze([16, 235])
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertEqual(result['summary']['supported_transition_count'], 0)
        self.assertEqual(result['candidates'], [])
        self.assertFalse(result['trace'][0]['baseline_supported'])
        single = self.analyze([16])
        self.assertEqual(single['status'], 'insufficient_evidence')
        self.assertIsNone(single['summary']['mean_change'])

    def test_continuous_spatial_motion_is_sustained_not_isolated(self):
        texture = np.random.default_rng(17).integers(0, 256, size=(16, 32), dtype=np.uint8)
        frames = np.stack([np.roll(texture, index, axis=1) for index in range(20)])
        result = analyze_frames(frames, np.arange(20) / 30, Config())
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['sustained_change_runs'][0]['transition_count'], 19)

    def test_timestamp_gaps_are_preserved_and_invalid_times_rejected(self):
        frames = np.zeros((3, 4, 4), dtype=np.uint8)
        result = analyze_frames(frames, [2, 2.1, 2.7], Config())
        self.assertAlmostEqual(result['trace'][1]['interval_seconds'], 0.6)
        for timestamps in ([0, 0, 1], [0, float('nan'), 1], [0, 1]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                analyze_frames(frames, timestamps, Config())

    def test_invalid_budgets_rejected(self):
        for options in ({'max_frames': 0}, {'width': 10000}, {'timeout': float('nan')},
                        {'max_frames': True}, {'baseline_radius': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options)

    def test_timeout_and_output_limit(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'], 0.1, 100)
        self.assertLess(time.monotonic() - started, 3)
        with self.assertRaises(ValueError):
            run_bounded([sys.executable, '-c', 'print("x" * 10000)'], 5, 100)
        with self.assertRaises(ValueError):
            run_bounded([sys.executable, '-c', 'import sys; sys.stderr.write("x" * 10000)'],
                        5, 100, stderr_limit=100)


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
class DecoderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clip = self.root / 'clip.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=64x48:rate=10:duration=2', '-c:v', 'mpeg4',
                        str(self.clip)], check=True, timeout=15, capture_output=True)

    def test_full_prefix_and_exact_cap(self):
        for cap, expected in ((21, 'full'), (20, 'full'), (7, 'prefix')):
            with self.subTest(cap=cap):
                result = measure(self.clip, Config(max_frames=cap))
                self.assertEqual(result['coverage']['kind'], expected)
                count = min(cap, 20)
                self.assertEqual(result['coverage']['frame_count'], count)
                self.assertEqual(len(result['trace']), count - 1)
                self.assertAlmostEqual(result['trace'][0]['timestamp_seconds'], 0.1)
                self.assertEqual(result['media']['sha256'],
                                 hashlib.sha256(self.clip.read_bytes()).hexdigest())
                self.assertIn('ffmpeg_version', result['provenance'])

    def cli(self, *arguments):
        return subprocess.run([sys.executable, '-m', 'dbmetrics.cli.scene_continuity', *map(str, arguments)],
                              capture_output=True, text=True, timeout=30)

    def test_json_resume_and_policy_conflict_do_not_overwrite(self):
        output = self.root / 'result.json'
        arguments = ('--clip', self.clip, '--output', output, '--max-frames', 7)
        self.assertEqual(self.cli(*arguments).returncode, 0)
        original = output.read_bytes()
        self.assertEqual(self.cli(*arguments).returncode, 0)
        self.assertEqual(output.read_bytes(), original)
        self.assertNotEqual(self.cli('--clip', self.clip, '--output', output,
                                    '--max-frames', 8).returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_manifest_resume_and_media_conflict(self):
        manifest = self.root / 'shard.jsonl'
        manifest.write_text(json.dumps({'clip_id': 'one', 'media_path': 'clip.mp4'}) + '\n')
        output = self.root / 'result.jsonl'
        arguments = ('--manifest', manifest, '--output', output)
        self.assertEqual(self.cli(*arguments).returncode, 0)
        original = output.read_bytes()
        self.assertEqual(self.cli(*arguments).returncode, 0)
        self.assertEqual(output.read_bytes(), original)
        self.clip.write_bytes(self.clip.read_bytes() + b'changed')
        self.assertNotEqual(self.cli(*arguments).returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_invalid_media_is_not_successful_full_coverage(self):
        self.clip.write_bytes(b'not video')
        with self.assertRaises(ValueError):
            measure(self.clip)

    def test_source_prefix_is_not_promoted_to_full_at_file_eof(self):
        result = measure(self.clip, Config(media_scope='prefix'))
        self.assertEqual(result['coverage']['kind'], 'full')
        self.assertEqual(result['coverage']['source_scope'], 'prefix')

    def test_encoded_step_is_localized_without_frame_sampling(self):
        clip = self.root / 'step.mp4'
        raw = np.broadcast_to(np.array([16] * 10 + [235] * 10, dtype=np.uint8)[:, None, None],
                              (20, 48, 64)).tobytes()
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'gray',
                        '-video_size', '64x48', '-framerate', '10', '-i', 'pipe:0',
                        '-c:v', 'mpeg4', str(clip)], input=raw, check=True,
                       timeout=15, capture_output=True)
        result = measure(clip)
        self.assertEqual(result['coverage']['frame_count'], 20)
        self.assertEqual([row['timestamp_seconds'] for row in result['candidates']], [1.0])

    def test_nonzero_variable_timestamps_are_not_replaced_with_uniform_samples(self):
        clip = self.root / 'variable.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=64x48:rate=10:duration=1', '-vf',
                        'setpts=PTS+2/TB+if(gte(N\\,5)\\,0.3/TB\\,0)',
                        '-fps_mode', 'passthrough', '-c:v', 'mpeg4', str(clip)],
                       check=True, timeout=15, capture_output=True)
        result = measure(clip)
        self.assertEqual(result['coverage']['frame_count'], 10)
        self.assertAlmostEqual(result['coverage']['first_timestamp_seconds'], 2)
        self.assertAlmostEqual(result['trace'][4]['interval_seconds'], 0.4)

    def test_shard_resumes_after_a_clip_failure(self):
        manifest = self.root / 'shard.jsonl'
        manifest.write_text('\n'.join(json.dumps({'clip_id': identifier, 'media_path': path})
                                     for identifier, path in [('one', 'clip.mp4'),
                                                              ('two', 'missing.mp4')]) + '\n')
        output = self.root / 'result.jsonl'
        arguments = ('--manifest', manifest, '--output', output)
        self.assertNotEqual(self.cli(*arguments).returncode, 0)
        original = output.read_bytes()
        self.assertEqual(len(original.splitlines()), 1)
        shutil.copyfile(self.clip, self.root / 'missing.mp4')
        self.assertEqual(self.cli(*arguments).returncode, 0)
        self.assertTrue(output.read_bytes().startswith(original))
        self.assertEqual(len(output.read_bytes().splitlines()), 2)
        self.assertEqual(self.cli(*arguments).returncode, 0)

    def test_partial_tail_and_changed_manifest_are_never_overwritten(self):
        manifest = self.root / 'shard.jsonl'
        manifest.write_text(json.dumps({'clip_id': 'one', 'media_path': 'clip.mp4'}) + '\n')
        output = self.root / 'result.jsonl'
        arguments = ('--manifest', manifest, '--output', output)
        self.assertEqual(self.cli(*arguments).returncode, 0)
        original = output.read_bytes()
        manifest.write_text(manifest.read_text() + '\n')
        self.assertNotEqual(self.cli(*arguments).returncode, 0)
        self.assertEqual(output.read_bytes(), original)
        output.write_bytes(original + b'{')
        self.assertNotEqual(self.cli(*arguments).returncode, 0)
        self.assertEqual(output.read_bytes(), original + b'{')


if __name__ == '__main__':
    unittest.main()
