import hashlib
from pathlib import Path

import pytest

from rnnoise_mlx.tools.save_openslr_resource_documents import save_resources


def test_save_resources_records_pages_and_linked_documents(tmp_path: Path):
    page = b'<a href="/resources/40/LICENSE">LICENSE</a>'
    license_body = b"license"

    def fetcher(url):
        body = page if url.endswith("/40/") else license_body
        return body, url, "text/plain"

    output = tmp_path / "legal"
    manifest = save_resources(["40"], output, fetcher=fetcher)

    assert manifest["resource_ids"] == ["40"]
    assert (output / "slr40/dataset-page.html").read_bytes() == page
    assert (output / "slr40/LICENSE").read_bytes() == license_body
    assert manifest["documents"][1]["sha256"] == hashlib.sha256(license_body).hexdigest()


def test_save_resources_rejects_duplicate_ids(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate"):
        save_resources(["40", "40"], tmp_path / "legal")
