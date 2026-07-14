from email.message import Message

from rnnoise_mlx.tools.preflight_corpus_archives import archive_urls, inspect_url, preflight


class Response:
    status = 200

    def __init__(self, url, length="123"):
        self._url = url
        self.headers = Message()
        self.headers["Content-Length"] = length
        self.headers["ETag"] = '"abc"'

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def geturl(self):
        return self._url


def test_archive_urls_are_unique_and_sorted():
    plan = {
        "official_sources": [{"archives": ["https://b", "https://a"]}],
        "regional_english_sources": [{"archives": ["https://a"]}],
    }
    assert archive_urls(plan) == ["https://a", "https://b"]


def test_inspect_url_records_headers_without_get():
    seen = {}

    def opener(request, timeout):
        seen["method"] = request.method
        seen["timeout"] = timeout
        return Response(request.full_url)

    result = inspect_url("https://example.test/a.zip", timeout=7, opener=opener)
    assert seen == {"method": "HEAD", "timeout": 7}
    assert result["content_length_bytes"] == 123
    assert result["etag"] == '"abc"'


def test_preflight_summarizes_known_sizes():
    plan = {
        "official_sources": [{"archives": ["https://b", "https://a"]}],
        "regional_english_sources": [],
    }

    def inspector(url, timeout):
        return {
            "url": url,
            "status": 200,
            "content_length_bytes": 5,
            "error": None,
        }

    result = preflight(plan, workers=2, inspector=inspector)
    assert result["archive_count"] == 2
    assert result["successful_count"] == 2
    assert result["sized_count"] == 2
    assert result["known_total_bytes"] == 10
    assert [item["url"] for item in result["archives"]] == ["https://a", "https://b"]
