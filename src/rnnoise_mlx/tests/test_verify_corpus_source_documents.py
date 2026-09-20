import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rnnoise_mlx.tools.verify_corpus_source_documents import sha256, verify


def fixture(root: Path):
    root.mkdir()
    (root / "rnnoise-datasets.txt").write_bytes(b"upstream")
    (root / "LICENSE").write_bytes(b"license")
    documents = []
    for name in ("rnnoise-datasets.txt", "LICENSE"):
        path = root / name
        documents.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"stage_id": "stage1", "documents": documents}
    (root / "manifest.json").write_text(json.dumps(manifest))
    plan = {"upstream": {"sha256": hashlib.sha256(b"upstream").hexdigest()}}
    review = {
        "evidence_manifest_sha256": sha256(root / "manifest.json"),
        "resources": [{"license_path": "LICENSE", "license_sha256": hashlib.sha256(b"license").hexdigest()}],
    }
    return plan, review


class VerifyCorpusSourceDocumentsTests(unittest.TestCase):
    def test_verify_checks_documents_plan_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            plan, review = fixture(docs)
            self.assertEqual(verify(docs, plan=plan, review=review), {
                "stage_id": "stage1",
                "document_count": 2,
                "reviewed_resource_count": 1,
            })


    def test_verify_rejects_changed_document(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            plan, review = fixture(docs)
            (docs / "LICENSE").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "LICENSE"):
                verify(docs, plan=plan, review=review)
