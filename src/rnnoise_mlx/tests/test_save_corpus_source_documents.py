from pathlib import Path

from rnnoise_mlx.tools.save_corpus_source_documents import document_links, save_documents, stage_resource_ids


def plan():
    return {
        "upstream": {"url": "https://example.test/datasets.txt"},
        "official_sources": [
            {"id": "a", "archives": ["https://www.openslr.org/resources/32/a.zip"]},
            {"id": "b", "archives": ["https://www.openslr.org/resources/37/b.zip"]},
        ],
        "regional_english_sources": [],
        "acquisition_stages": [{"id": "stage1", "source_ids": ["a", "b"]}],
    }


def test_stage_resource_ids_are_unique_and_numeric_sorted():
    assert stage_resource_ids(plan(), "stage1") == ["32", "37"]


def test_document_links_keep_small_metadata_names_for_same_resource():
    html = b'<a href="https://mirror.test/resources/32/LICENSE.txt">license</a><a href="/resources/32/audio.zip">audio</a><a href="/resources/37/README.txt">other</a>'
    assert document_links("https://www.openslr.org/32/", html) == [
        "https://www.openslr.org/resources/32/LICENSE.txt"
    ]


def test_document_links_recovers_unlinked_license_name():
    html = b"See LICENSE.txt file for license information."
    assert document_links("https://www.openslr.org/32/", html) == [
        "https://www.openslr.org/resources/32/LICENSE.txt"
    ]


def test_save_documents_writes_checksummed_manifest(tmp_path: Path):
    pages = {
        "https://example.test/datasets.txt": b"upstream",
        "https://www.openslr.org/32/": b'<a href="/resources/32/LICENSE.txt">license</a>',
        "https://www.openslr.org/resources/32/LICENSE.txt": b"license32",
        "https://www.openslr.org/37/": b'<a href="/resources/37/README.txt">readme</a>',
        "https://www.openslr.org/resources/37/README.txt": b"readme37",
    }

    def fetcher(url):
        return pages[url], url, "text/plain"

    manifest = save_documents(plan(), "stage1", tmp_path / "legal", fetcher=fetcher)
    assert len(manifest["documents"]) == 5
    assert (tmp_path / "legal/slr32/LICENSE.txt").read_bytes() == b"license32"
    assert all(len(record["sha256"]) == 64 for record in manifest["documents"])
