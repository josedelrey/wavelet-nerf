import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile


class ExampleDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repository = self.root / "repo with spaces"
        self.repository.mkdir()
        self.script = self.repository / "download_dataset.sh"
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / "download_dataset.sh", self.script
        )
        self.archive = self.root / "example.zip"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        wget = self.bin / "wget"
        wget.write_text(
            "#!/usr/bin/env python3\n"
            "import os, pathlib, shutil, sys\n"
            'destination = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--output-document="))\n'
            'shutil.copyfile(os.environ["TEST_ARCHIVE"], destination)\n'
            'with pathlib.Path(os.environ["TEST_DOWNLOADS"]).open("a") as file: file.write("download\\n")\n'
        )
        wget.chmod(0o755)
        self.calls = self.root / "downloads.txt"

    def make_archive(self, fern=True):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for split in ("train", "val", "test"):
                archive.writestr(f"nerf_synthetic/lego/transforms_{split}.json", "{}")
                archive.writestr(
                    f"nerf_synthetic/lego/{split}/image.png", "example image"
                )
            if fern:
                archive.writestr(
                    "nerf_llff_data/fern/poses_bounds.npy", "example camera metadata"
                )
                archive.writestr(
                    "nerf_llff_data/fern/images/image.JPG", "example image"
                )
                archive.writestr(
                    "nerf_llff_data/fern/images_8/image.png", "cached example image"
                )

    def run_download(self):
        return subprocess.run(
            ["bash", str(self.script)],
            cwd=self.root,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "TEST_ARCHIVE": str(self.archive),
                "TEST_DOWNLOADS": str(self.calls),
            },
        )

    def test_installs_both_scenes_and_caches_from_another_working_directory(self):
        self.make_archive()
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stderr)
        datasets = self.repository / "datasets"
        self.assertTrue((datasets / "lego/transforms_train.json").is_file())
        self.assertTrue((datasets / "fern/poses_bounds.npy").is_file())
        self.assertTrue((datasets / "fern/images_8/image.png").is_file())
        self.assertEqual(list(datasets.glob(".nerf-download.*")), [])
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls.read_text().splitlines(), ["download"])

    def test_existing_scene_is_preserved_when_installing_the_missing_scene(self):
        self.make_archive()
        fern = self.repository / "datasets/fern"
        fern.mkdir(parents=True)
        sentinel = fern / "my-data.txt"
        sentinel.write_text("keep this")
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sentinel.read_text(), "keep this")
        self.assertFalse((fern / "poses_bounds.npy").exists())
        self.assertTrue(
            (self.repository / "datasets/lego/transforms_test.json").is_file()
        )

    def test_invalid_archive_installs_nothing_and_cleans_temporary_files(self):
        self.make_archive(fern=False)
        result = self.run_download()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list((self.repository / "datasets").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
