import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from dbmetrics import source_statistics
from dbmetrics.cli import youtube_sources


SOURCE_ID = "--42OoF8WJM"
OBSERVED_AT = "2026-09-30T23:30:00Z"


class SourceStatisticsTests(unittest.TestCase):
    def test_catalog_is_read_only_and_preserves_unknown_snapshot_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE sources (source_id TEXT PRIMARY KEY, channel_id TEXT, views INTEGER)")
                database.execute("INSERT INTO sources VALUES (?, ?, ?)", (SOURCE_ID, "channel", 100))
            database.close()
            before = path.read_bytes()
            rows = source_statistics.read_catalog(path)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(rows, [{"source_id": SOURCE_ID, "channel_id": "channel", "views": 100}])
        record = source_statistics.catalog_record(rows[0], catalog_sha256="a" * 64)
        self.assertEqual(record["metrics"]["source_view_count"], 100)
        self.assertIsNone(record["metrics"]["source_like_count"])
        self.assertIsNone(record["metrics"]["source_comment_count"])
        self.assertIsNone(record["observed_at"])
        self.assertEqual(record["scope"], "source_video")
        self.assertEqual(record["provenance"]["catalog_sha256"], "a" * 64)

    def test_youtube_identity_counts_and_upload_date(self):
        record = source_statistics.youtube_record(
            SOURCE_ID, {"id": SOURCE_ID, "channel_id": "channel", "view_count": 100,
                        "like_count": 0, "comment_count": 2, "upload_date": "20231117"},
            observed_at=OBSERVED_AT, extractor_version="2026.08.19")
        self.assertEqual(record["metrics"], {"source_view_count": 100, "source_like_count": 0,
                                            "source_comment_count": 2})
        self.assertEqual(record["uploaded_on"], "2023-11-17")
        self.assertEqual(record["observed_at"], OBSERVED_AT)
        self.assertEqual(record["provenance"]["extractor_version"], "2026.08.19")
        self.assertNotIn("formats", json.dumps(record))

    def test_missing_counts_are_unknown_not_zero(self):
        record = source_statistics.youtube_record(
            SOURCE_ID, {"id": SOURCE_ID, "view_count": 0},
            observed_at=OBSERVED_AT, extractor_version="test")
        self.assertEqual(record["metrics"]["source_view_count"], 0)
        self.assertIsNone(record["metrics"]["source_like_count"])
        self.assertIsNone(record["metrics"]["source_comment_count"])
        self.assertIsNone(record["uploaded_on"])

    def test_invalid_counts_ids_dates_and_timestamps_are_rejected(self):
        for value in (True, -1, 1.5, "12", float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                source_statistics.youtube_record(
                    SOURCE_ID, {"id": SOURCE_ID, "like_count": value},
                    observed_at=OBSERVED_AT, extractor_version="test")
        for changes in ({"id": "different"}, {"upload_date": "20230230"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                source_statistics.youtube_record(
                    SOURCE_ID, {"id": SOURCE_ID, **changes},
                    observed_at=OBSERVED_AT, extractor_version="test")
        for timestamp in ("2026-09-30", "2026-09-30T23:30:00", "bad"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                source_statistics.youtube_record(
                    SOURCE_ID, {"id": SOURCE_ID}, observed_at=timestamp, extractor_version="test")
        with self.assertRaises(ValueError):
            source_statistics.youtube_record(
                "https://example.com", {"id": "https://example.com"},
                observed_at=OBSERVED_AT, extractor_version="test")

    def test_each_source_is_exported_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE sources (source_id TEXT PRIMARY KEY, channel_id TEXT, views INTEGER)")
                database.execute("CREATE TABLE clips (clip_id TEXT, source_id TEXT)")
                database.execute("INSERT INTO sources VALUES (?, ?, ?)", (SOURCE_ID, "channel", 100))
                database.executemany("INSERT INTO clips VALUES (?, ?)", [("part1", SOURCE_ID), ("part2", SOURCE_ID)])
            database.close()
            self.assertEqual(len(source_statistics.read_catalog(path)), 1)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = Path(self.directory.name) / "catalog.sqlite3"
        self.output = Path(self.directory.name) / "results.jsonl"
        dependency = patch.dict(sys.modules, {"yt_dlp": object()})
        dependency.start()
        self.addCleanup(dependency.stop)
        with sqlite3.connect(self.catalog) as database:
            database.execute("CREATE TABLE sources (source_id TEXT PRIMARY KEY, channel_id TEXT, views INTEGER)")
            database.execute("INSERT INTO sources VALUES (?, ?, ?)", (SOURCE_ID, "channel", 100))
        database.close()

    def command(self, provider="catalog", *extra):
        return ["--catalog", str(self.catalog), "--output", str(self.output), "--provider", provider, *extra]

    def test_catalog_export_and_repeat_do_not_duplicate_records(self):
        self.assertEqual(youtube_sources.main(self.command()), 0)
        first = self.output.read_bytes()
        self.assertEqual(youtube_sources.main(self.command()), 0)
        self.assertEqual(self.output.read_bytes(), first)
        self.assertEqual(json.loads(first)["metrics"]["source_view_count"], 100)

    def test_python_api_called_once_per_source_and_resume_skips_it(self):
        with patch.object(youtube_sources, "fetch_youtube", return_value=(
                {"id": SOURCE_ID, "like_count": 7}, "test")) as fetch:
            self.assertEqual(youtube_sources.main(self.command("youtube", "--delay", "0")), 0)
            self.assertEqual(youtube_sources.main(self.command("youtube", "--delay", "0")), 0)
        fetch.assert_called_once_with(SOURCE_ID)
        self.assertEqual(json.loads(self.output.read_text())["metrics"]["source_like_count"], 7)

    def test_failed_lookup_is_not_fabricated_as_zero(self):
        with patch.object(youtube_sources, "fetch_youtube", side_effect=RuntimeError("Video unavailable")):
            self.assertEqual(youtube_sources.main(self.command("youtube", "--delay", "0")), 1)
        record = json.loads(self.output.read_text())
        self.assertEqual(record["status"], "error")
        self.assertTrue(all(value is None for value in record["metrics"].values()))

    def test_mixed_provider_and_truncated_resume_are_rejected(self):
        youtube_sources.main(self.command())
        with self.assertRaises(ValueError):
            youtube_sources.main(self.command("youtube"))
        self.output.write_text('{"source_id":')
        with self.assertRaises(ValueError):
            youtube_sources.main(self.command())

    def test_negative_delay_and_limit_are_rejected_before_output(self):
        for options in (("--delay", "-1"), ("--limit", "0"), ("--delay", "nan")):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                youtube_sources.main(self.command("catalog", *options))
        self.assertFalse(self.output.exists())

    def test_access_limit_stops_before_requesting_another_source(self):
        with sqlite3.connect(self.catalog) as database:
            database.execute("INSERT INTO sources VALUES (?, ?, ?)", ("abcdefghijk", "channel", 1))
        database.close()
        with patch.object(youtube_sources, "fetch_youtube", side_effect=RuntimeError("HTTP Error 429")) as fetch:
            self.assertEqual(youtube_sources.main(self.command("youtube", "--all", "--delay", "0")), 1)
        fetch.assert_called_once_with(SOURCE_ID)
        self.assertEqual(len(self.output.read_text().splitlines()), 1)

    def test_concurrent_collector_cannot_duplicate_source_requests(self):
        with self.output.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "Another collector"):
                youtube_sources.main(self.command())

    def test_shards_cover_sources_once_and_skip_shared_checkpoint(self):
        identities = [SOURCE_ID, 'abcdefghijk', '01234567890', 'ABCDEFGHIJK']
        with sqlite3.connect(self.catalog) as database:
            database.executemany('INSERT INTO sources VALUES (?, ?, ?)',
                                 [(identity, 'channel', 1) for identity in identities[1:]])
        checkpoint = Path(self.directory.name) / 'checkpoint.jsonl'
        youtube_sources.main(self.command('catalog', '--limit', '1'))
        checkpoint.write_bytes(self.output.read_bytes())
        collected = []
        for shard in range(4):
            self.output = Path(self.directory.name) / f'shard-{shard}.jsonl'
            youtube_sources.main(self.command('catalog', '--all', '--shard-count', '4',
                                             '--shard-index', str(shard), '--resume-from', str(checkpoint)))
            collected.extend(json.loads(line)['source_id'] for line in self.output.read_text().splitlines())
        prior = json.loads(checkpoint.read_text())['source_id']
        self.assertCountEqual(collected, [identity for identity in identities if identity != prior])

    def test_shared_stop_prevents_other_shards_from_fetching(self):
        stop = Path(self.directory.name) / 'stop'
        stop.write_text('HTTP Error 429')
        with patch.object(youtube_sources, 'fetch_youtube') as fetch:
            self.assertEqual(youtube_sources.main(self.command('youtube', '--stop-file', str(stop))), 1)
        fetch.assert_not_called()

    def test_request_clock_spaces_request_starts(self):
        clock = Path(self.directory.name) / 'request-clock'
        clock.write_text('100')
        with patch.object(youtube_sources.time, 'time', side_effect=[101, 102]), \
                patch.object(youtube_sources.time, 'sleep') as sleep:
            youtube_sources.wait_for_request(clock, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(float(clock.read_text()), 102)


if __name__ == "__main__":
    unittest.main()
