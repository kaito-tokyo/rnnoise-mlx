from pathlib import Path
import tempfile
import unittest

from rnnoise_mlx.tools.split_audio_directory import assignment, split


class SplitAudioDirectoryTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_validates_fraction(self):
        self.assertEqual(assignment("a.flac", 0.1, 141), assignment("a.flac", 0.1, 141))
        with self.assertRaises(ValueError):
            assignment("a.flac", 0, 141)


    def test_split_creates_symlinks_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            probe = tmp_path / "symlink-probe"
            try:
                probe.symlink_to(tmp_path)
            except OSError as error:
                self.skipTest(f"symlink creation is unavailable: {error}")
            source = tmp_path / "source"
            source.mkdir()
            for index in range(20):
                (source / f"{index}.flac").write_bytes(b"audio")
            output = tmp_path / "split"
            output.mkdir()
            manifest = split(source, output, 0.5, 141)
            self.assertEqual(sum(manifest["counts"].values()), 20)
            self.assertGreater(manifest["counts"]["train"], 0)
            self.assertGreater(manifest["counts"]["eval"], 0)
            self.assertEqual(len(list(output.rglob("*.flac"))), 20)
            self.assertTrue(all(path.is_symlink() for path in output.rglob("*.flac")))
            self.assertTrue((output / "split-manifest.json").is_file())
