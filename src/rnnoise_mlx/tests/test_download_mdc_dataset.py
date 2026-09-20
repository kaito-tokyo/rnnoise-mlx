import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rnnoise_mlx.tools.download_mdc_dataset import stream_download


class Response(io.BytesIO):
    def __init__(self, payload, status):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class StreamDownloadTests(unittest.TestCase):
    def test_stream_download_resumes_with_range(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "archive.partial"
            partial.write_bytes(b"abc")

            def open_request(request, timeout):
                self.assertEqual(request.headers["Range"], "bytes=3-")
                self.assertEqual(timeout, 120)
                return Response(b"def", 206)

            with patch("urllib.request.urlopen", open_request):
                stream_download("https://storage.example/archive", partial, 6)
            self.assertEqual(partial.read_bytes(), b"abcdef")


    def test_stream_download_rejects_ignored_range(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "archive.partial"
            partial.write_bytes(b"abc")
            with patch("urllib.request.urlopen", lambda *_args, **_kwargs: Response(b"abcdef", 200)):
                with self.assertRaisesRegex(RuntimeError, "Range"):
                    stream_download("https://storage.example/archive", partial, 6)


    def test_stream_download_reconnects_after_transient_read_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "archive.partial"
            requests = []

            class BrokenResponse(Response):
                def read(self, _size=-1):
                    raise OSError("transient TLS failure")

            def open_request(request, timeout):
                requests.append(request.headers.get("Range"))
                if len(requests) == 1:
                    return BrokenResponse(b"", 200)
                return Response(b"abcdef", 200)

            with patch("urllib.request.urlopen", open_request), patch("time.sleep"):
                stream_download("https://storage.example/archive", partial, 6)

            self.assertEqual(requests, [None, None])
            self.assertEqual(partial.read_bytes(), b"abcdef")


    def test_stream_download_reconnects_after_early_eof_with_range(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "archive.partial"
            responses = iter((Response(b"abc", 200), Response(b"def", 206)))
            ranges = []

            def open_request(request, timeout):
                ranges.append(request.headers.get("Range"))
                return next(responses)

            with patch("urllib.request.urlopen", open_request), patch("time.sleep"):
                stream_download("https://storage.example/archive", partial, 6)

            self.assertEqual(ranges, [None, "bytes=3-"])
            self.assertEqual(partial.read_bytes(), b"abcdef")
