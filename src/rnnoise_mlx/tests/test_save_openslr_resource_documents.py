import hashlib
import tempfile
import unittest
from pathlib import Path

from rnnoise_mlx.tools.save_openslr_resource_documents import save_resources


class SaveOpenSLRResourceDocumentsTests(unittest.TestCase):
    def test_save_resources_records_pages_and_linked_documents(self):
        page = b'<a href="/resources/40/LICENSE">LICENSE</a>'
        license_body = b"license"

        def fetcher(url):
            body = page if url.endswith("/40/") else license_body
            return body, url, "text/plain"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "legal"
            manifest = save_resources(["40"], output, fetcher=fetcher)

            self.assertEqual(manifest["resource_ids"], ["40"])
            self.assertEqual(
                (output / "slr40/dataset-page.html").read_bytes(), page
            )
            self.assertEqual((output / "slr40/LICENSE").read_bytes(), license_body)
            self.assertEqual(
                manifest["documents"][1]["sha256"],
                hashlib.sha256(license_body).hexdigest(),
            )


    def test_save_resources_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                save_resources(["40", "40"], Path(directory) / "legal")
