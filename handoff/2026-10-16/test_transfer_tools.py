"""Small standard-library checks for transfer integrity, not numerical tests."""

import hashlib
import importlib.util
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


verifier = module("verify_export")
exporter = module("export_bundle")


class TransferTests(unittest.TestCase):
    def build(
        self,
        root,
        *,
        digest=None,
        extra=False,
        duplicate=False,
        name="bundle/data.txt",
        symlink=False,
        stable=True,
    ):
        path = root / "sample.zip"
        value = b"preserved evidence\n"
        row = {
            "archive_path": name,
            "bytes": len(value),
            "sha256": digest or hashlib.sha256(value).hexdigest(),
            "stable_during_copy": stable,
        }
        with zipfile.ZipFile(path, "x") as archive:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, value)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(name, value)
            if extra:
                archive.writestr("bundle/unlisted.txt", b"unlisted")
            archive.writestr(
                "bundle/MANIFEST.json",
                json.dumps({"schema": "omnibias-research-transfer/v1", "files": [row]}),
            )
        return path

    def test_valid_full_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            result = verifier.verify(self.build(Path(folder)))
            self.assertTrue(result["integrity_passed"])
            self.assertTrue(result["snapshot_origin_consistency_passed"])
            self.assertFalse(result["numerical_or_benchmark_validation"])

    def test_wrong_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Checksum"):
                verifier.verify(self.build(Path(folder), digest="0" * 64))

    def test_unlisted_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "membership"):
                verifier.verify(self.build(Path(folder), extra=True))

    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verifier.verify(self.build(Path(folder), duplicate=True))

    def test_unsafe_paths(self):
        for name in ["../escape", "/absolute", "a/../b", "a/./b", "a//b", "a\\b", "C:escape"]:
            self.assertFalse(verifier.safe_name(name), name)

    def test_symlink_payload_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "type"):
                verifier.verify(self.build(Path(folder), symlink=True))

    def test_unstable_origin_not_promoted(self):
        with tempfile.TemporaryDirectory() as folder:
            result = verifier.verify(self.build(Path(folder), stable=False))
            self.assertTrue(result["integrity_passed"])
            self.assertFalse(result["snapshot_origin_consistency_passed"])

    def test_live_cutoff(self):
        self.assertFalse(
            exporter.excluded_live(Path("completion_execution_20261013/matched/00009.npz"), 10)
        )
        self.assertTrue(
            exporter.excluded_live(
                Path("completion_execution_20261013/matched/00010.monitor.json"), 10
            )
        )
        self.assertFalse(
            exporter.excluded_live(Path("reuse_execution_20261012/matched/00010.json"), 10)
        )

    def test_process_manifest_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_root = root / "completion_execution_20261013"
            source_root.mkdir()
            source = source_root / "protocol_process_manifest.jsonl"
            source.write_bytes(b'{"index":0}\n{"index":1}\n{"index":2}\n{"partial":')
            archive_path = root / "copy.zip"
            with zipfile.ZipFile(archive_path, "x") as archive:
                row = exporter.copy_file(archive, source, "bundle/log.jsonl", cutoff=2)
            self.assertTrue(row["stable_during_copy"])
            self.assertEqual(row["omitted_process_manifest_rows"], 2)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.read("bundle/log.jsonl"), b'{"index":0}\n{"index":1}\n')


if __name__ == "__main__":
    unittest.main()
