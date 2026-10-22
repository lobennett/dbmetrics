"""Source-video counts; normalization is offline and never downloads media."""

from datetime import date, datetime
from pathlib import Path
import re
import sqlite3


METRICS = {
    "source_view_count": "Source views",
    "source_like_count": "Source likes",
    "source_comment_count": "Source comments",
}


def _identity(source_id):
    if type(source_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{11}", source_id):
        raise ValueError("Expected a YouTube source-video ID, not a clip ID or URL")
    return source_id


def _count(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("Counts must be nonnegative integers or null")
    return value


def _record(source_id, channel_id, counts, *, observed_at, provenance):
    return {
        "schema_version": "dbp-source-statistics-v1",
        "source_id": _identity(source_id),
        "scope": "source_video",
        "channel_id": channel_id,
        "metrics": dict(zip(METRICS, map(_count, counts))),
        "uploaded_on": None,
        "observed_at": observed_at,
        "status": "ok",
        "provenance": provenance,
    }


def read_catalog(path: str | Path) -> list[dict]:
    """Read one row per source; clip rows never multiply requests."""
    path = Path(path).resolve(strict=True)
    database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in database.execute(
            "SELECT source_id, channel_id, views FROM sources ORDER BY source_id")]
    finally:
        database.close()


def catalog_record(row: dict, *, catalog_sha256: str) -> dict:
    """The original collection date is unknown; filesystem dates are not substitutes."""
    return _record(row["source_id"], row["channel_id"], (row["views"], None, None),
                   observed_at=None, provenance={"provider": "catalog", "catalog_sha256": catalog_sha256})


def youtube_record(source_id: str, info: dict, *, observed_at: str, extractor_version: str) -> dict:
    """Normalize a metadata-only yt-dlp observation, preserving absent counts."""
    _identity(source_id)
    if info.get("id") != source_id:
        raise ValueError("Returned source identity differs from the requested source")
    timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if "T" not in observed_at or timestamp.utcoffset() is None:
        raise ValueError("observed_at must include a time and timezone")
    record = _record(source_id, info.get("channel_id"),
                     (info.get("view_count"), info.get("like_count"), info.get("comment_count")),
                     observed_at=observed_at,
                     provenance={"provider": "youtube", "extractor": "yt-dlp",
                                 "extractor_version": extractor_version, "download_media": False,
                                 "count_precision": "extractor_reported; exactness_unverified"})
    uploaded = info.get("upload_date")
    if uploaded is not None:
        if type(uploaded) is not str or not re.fullmatch(r"\d{8}", uploaded):
            raise ValueError("upload_date must have YYYYMMDD format")
        record["uploaded_on"] = date(int(uploaded[:4]), int(uploaded[4:6]), int(uploaded[6:])).isoformat()
    return record
