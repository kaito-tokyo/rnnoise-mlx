import io

import pytest

from rnnoise_mlx.tools.download_mdc_dataset import stream_download


class Response(io.BytesIO):
    def __init__(self, payload, status):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_stream_download_resumes_with_range(monkeypatch, tmp_path):
    partial = tmp_path / "archive.partial"
    partial.write_bytes(b"abc")

    def open_request(request, timeout):
        assert request.headers["Range"] == "bytes=3-"
        assert timeout == 120
        return Response(b"def", 206)

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    stream_download("https://storage.example/archive", partial, 6)
    assert partial.read_bytes() == b"abcdef"


def test_stream_download_rejects_ignored_range(monkeypatch, tmp_path):
    partial = tmp_path / "archive.partial"
    partial.write_bytes(b"abc")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response(b"abcdef", 200))
    with pytest.raises(RuntimeError, match="Range"):
        stream_download("https://storage.example/archive", partial, 6)


def test_stream_download_reconnects_after_transient_read_failure(monkeypatch, tmp_path):
    partial = tmp_path / "archive.partial"
    requests = []

    class BrokenResponse(Response):
        def read(self, _size=-1):
            raise OSError("transient TLS failure")

    def open_request(request, timeout):
        requests.append(request.headers.get("Range"))
        if len(requests) == 1:
            return BrokenResponse(b"", 200)
        return Response(b"abcdef", 200)

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    stream_download("https://storage.example/archive", partial, 6)

    assert requests == [None, None]
    assert partial.read_bytes() == b"abcdef"


def test_stream_download_reconnects_after_early_eof_with_range(monkeypatch, tmp_path):
    partial = tmp_path / "archive.partial"
    responses = iter((Response(b"abc", 200), Response(b"def", 206)))
    ranges = []

    def open_request(request, timeout):
        ranges.append(request.headers.get("Range"))
        return next(responses)

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    stream_download("https://storage.example/archive", partial, 6)

    assert ranges == [None, "bytes=3-"]
    assert partial.read_bytes() == b"abcdef"
