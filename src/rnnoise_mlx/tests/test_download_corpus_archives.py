import hashlib
import io
import urllib.error
from pathlib import Path

import pytest

from rnnoise_mlx.tools.download_corpus_archives import download, head, selected_archives


PLAN = {
    "official_sources": [{"id": "a", "archives": ["https://example.test/a.zip"]}],
    "regional_english_sources": [{"id": "b", "archives": ["https://example.test/b.zip"]}],
    "preferred_complement": {
        "sources": [{"id": "c", "archives": ["https://example.test/c.tar.gz"]}]
    },
}


def test_selected_archives_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        selected_archives(PLAN, ["missing"])


def test_selected_archives_includes_preferred_complement():
    assert selected_archives(PLAN, ["c"]) == [
        {"source_id": "c", "url": "https://example.test/c.tar.gz"}
    ]


def test_download_records_size_and_digest(tmp_path: Path):
    bodies = {"https://example.test/a.zip": b"archive-a"}

    def metadata(url):
        return {
            "content_length_bytes": len(bodies[url]),
            "etag": '"etag"',
            "last_modified": "date",
            "final_url": url,
        }

    def downloader(url, path):
        path.write_bytes(bodies[url])

    manifest = download(PLAN, ["a"], tmp_path / "archives", metadata_fetcher=metadata, downloader=downloader)
    assert manifest["total_bytes"] == len(bodies["https://example.test/a.zip"])
    assert manifest["archives"][0]["sha256"] == hashlib.sha256(b"archive-a").hexdigest()
    assert not (tmp_path / "archives/a.zip.part").exists()


def test_download_keeps_partial_on_size_mismatch(tmp_path: Path):
    def metadata(_url):
        return {"content_length_bytes": 10, "etag": None, "last_modified": None, "final_url": "x"}

    def downloader(_url, path):
        path.write_bytes(b"short")

    with pytest.raises(ValueError, match="size mismatch"):
        download(PLAN, ["a"], tmp_path / "archives", metadata_fetcher=metadata, downloader=downloader)
    assert (tmp_path / "archives/a.zip.part").read_bytes() == b"short"


def test_download_resumes_transaction_with_complete_and_partial_archives(tmp_path: Path):
    bodies = {
        "https://example.test/a.zip": b"archive-a",
        "https://example.test/b.zip": b"archive-b",
    }
    output_dir = tmp_path / "archives"
    output_dir.mkdir()
    (output_dir / "a.zip").write_bytes(bodies["https://example.test/a.zip"])
    (output_dir / "b.zip.part").write_bytes(b"archive-")
    downloaded = []

    def metadata(url):
        return {
            "content_length_bytes": len(bodies[url]),
            "etag": None,
            "last_modified": None,
            "final_url": url,
        }

    def downloader(url, path):
        downloaded.append(url)
        path.write_bytes(bodies[url])

    manifest = download(
        PLAN,
        ["a", "b"],
        output_dir,
        metadata_fetcher=metadata,
        downloader=downloader,
    )

    assert downloaded == ["https://example.test/b.zip"]
    assert [row["path"] for row in manifest["archives"]] == ["a.zip", "b.zip"]
    assert manifest["total_bytes"] == sum(map(len, bodies.values()))
    assert not (output_dir / "b.zip.part").exists()


def test_download_rejects_wrong_sized_existing_archive(tmp_path: Path):
    output_dir = tmp_path / "archives"
    output_dir.mkdir()
    (output_dir / "a.zip").write_bytes(b"wrong")

    def metadata(url):
        return {
            "content_length_bytes": len(b"archive-a"),
            "etag": None,
            "last_modified": None,
            "final_url": url,
        }

    with pytest.raises(ValueError, match="size mismatch for existing a.zip"):
        download(PLAN, ["a"], output_dir, metadata_fetcher=metadata)

    assert (output_dir / "a.zip").read_bytes() == b"wrong"


def test_head_falls_back_to_one_byte_range_when_head_is_forbidden(monkeypatch):
    requests = []

    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(b"x")
            self.headers = {
                "Content-Range": "bytes 0-0/1234",
                "ETag": '"etag"',
                "Last-Modified": "date",
            }

        def geturl(self):
            return "https://storage.example/final.zip"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def open_request(request, timeout):
        requests.append(request)
        if request.get_method() == "HEAD":
            raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, None)
        assert request.headers["Range"] == "bytes=0-0"
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    metadata = head("https://storage.example/archive.zip")

    assert [request.get_method() for request in requests] == ["HEAD", "GET"]
    assert metadata["content_length_bytes"] == 1234
    assert metadata["final_url"] == "https://storage.example/final.zip"
    assert metadata["filename"] == "final.zip"
