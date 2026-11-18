# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import webdataset as wds

from megatron.energon import Cooker, WorkerConfig, get_savable_loader, get_train_dataset, stateless
from megatron.energon.epathlib import EPath
from megatron.energon.epathlib.metadata_snapshot import (
    MetadataSnapshot,
    active_snapshot,
    capture_metadata,
    use_metadata_snapshot,
)
from megatron.energon.flavors import BaseWebdatasetFactory
from tests import test_jsonl_dataset


@stateless()
def cook_without_snapshot(sample):
    assert active_snapshot() is None, "construction cache leaked into worker"
    return test_jsonl_dataset.cook_text(sample)


class SnapshotWorkerEncoder(test_jsonl_dataset.SimpleCookingTaskEncoder):
    cookers = [Cooker(cook=cook_without_snapshot)]


class SnapshotTest(unittest.TestCase):
    def test_public_loader_order_and_resume_with_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            test_jsonl_dataset.TestJsonlDataset.create_text_test_dataset(
                source, range(32), range(32)
            )
            for num_workers in (0, 2):
                with self.subTest(num_workers=num_workers):
                    worker = WorkerConfig(rank=0, world_size=1, num_workers=num_workers)

                    def dataset():
                        return get_train_dataset(
                            source,
                            worker_config=worker,
                            batch_size=None,
                            shuffle_buffer_size=None,
                            max_samples_per_sequence=None,
                            task_encoder=test_jsonl_dataset.SimpleCookingTaskEncoder(),
                        )

                    package = root / "metadata.json"
                    with capture_metadata(package, recipe_identity="recipe"):
                        original = dataset()
                    snapshot = MetadataSnapshot.load(
                        package,
                        sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                        recipe_identity="recipe",
                    )
                    validated = snapshot.validate()
                    with use_metadata_snapshot(validated):
                        replayed = dataset()
                    with get_savable_loader(original) as baseline:
                        iterator = iter(baseline)
                        expected = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(16))
                        ]
                    with get_savable_loader(replayed) as loader:
                        iterator = iter(loader)
                        prefix = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(5))
                        ]
                        state = loader.save_state_rank()
                        self.assertEqual(prefix, expected[:5])
                    with use_metadata_snapshot(validated):
                        restored = dataset()
                    with get_savable_loader(restored) as loader:
                        loader.restore_state_rank(state)
                        iterator = iter(loader)
                        suffix = [
                            (sample.idx, sample.text)
                            for sample in (next(iterator) for _ in range(11))
                        ]
                        self.assertEqual(suffix, expected[5:])

    def test_capture_replay_without_storage_calls_and_scope_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recipe.yaml"
            source.write_text("dataset: αβ\n")
            path = EPath(source)
            missing = EPath(root / "absent.yaml")
            package = root / "metadata.json"
            with capture_metadata(package, recipe_identity="recipe-sha"):
                self.assertTrue(path.is_file())
                self.assertFalse(missing.is_file())
                expected_size = path.size()
                self.assertEqual(path.read_text(), source.read_text())
            snapshot = MetadataSnapshot.load(
                package,
                sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                recipe_identity="recipe-sha",
            )
            validated = snapshot.validate()
            with use_metadata_snapshot(validated):
                with (
                    patch.object(
                        type(path.fs),
                        "open",
                        side_effect=AssertionError("storage open"),
                    ),
                    patch.object(
                        type(path.fs),
                        "info",
                        side_effect=AssertionError("storage info"),
                    ),
                    patch.object(
                        type(path.fs),
                        "is_file",
                        side_effect=AssertionError("storage is_file"),
                    ),
                ):
                    self.assertTrue(path.is_file())
                    self.assertFalse(missing.is_file())
                    self.assertEqual(path.size(), expected_size)
                    with path.open("rb") as first, path.open("rb") as second:
                        self.assertEqual(first.read(3), b"dat")
                        self.assertEqual(second.read(), source.read_bytes())
                    self.assertEqual(path.read_text(), source.read_text())
            source.write_text("new dataset revision\n")
            self.assertEqual(path.read_text(), "new dataset revision\n")

    def test_digest_and_recipe_identity_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "metadata.json"
            with capture_metadata(package, recipe_identity="recipe-sha"):
                pass
            digest = hashlib.sha256(package.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "SHA256"):
                MetadataSnapshot.load(package, sha256="wrong", recipe_identity="recipe-sha")
            with self.assertRaisesRegex(ValueError, "identity"):
                MetadataSnapshot.load(package, sha256=digest, recipe_identity="wrong")

    def test_stale_recipe_jsonl_and_index_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            recipe = root / "recipe.yaml"
            recipe.write_text("version: first\n")

            def prepare(count):
                test_jsonl_dataset.TestJsonlDataset.create_text_test_dataset(
                    source, range(count), range(count)
                )

            def dataset():
                return get_train_dataset(
                    source,
                    worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    task_encoder=test_jsonl_dataset.SimpleCookingTaskEncoder(),
                )

            prepare(10)
            package = root / "metadata.json"
            with capture_metadata(package, recipe_identity="revision"):
                dataset()
                EPath(recipe).read_text()
            snapshot = MetadataSnapshot.load(
                package,
                sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                recipe_identity="revision",
            )
            with source.open("a") as stream:
                stream.write('{"idx":10,"txt":"10"}\n')
            with self.assertRaisesRegex(AssertionError, "index.*does not match"):
                dataset()
            with self.assertRaisesRegex(ValueError, "Stale"):
                snapshot.validate()
            prepare(30)
            with self.assertRaisesRegex(ValueError, "Stale"):
                snapshot.validate()
            prepare(10)
            recipe.write_text("version: other\n")
            with self.assertRaisesRegex(ValueError, "contents changed"):
                snapshot.validate()
            recipe.write_text("version: first\n")
            # Same index length, different terminal offset: size checks alone cannot catch it.
            index = source.with_suffix(".jsonl.idx")
            original = index.read_bytes()
            index.write_bytes(original[:-8] + (1).to_bytes(8, "little"))
            with self.assertRaisesRegex(ValueError, "JSONL index changed"):
                snapshot.validate()

    def test_webdataset_generation_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with wds.TarWriter(str(root / "data.tar")) as writer:
                for index in range(8):
                    writer.write({"__key__": str(index), "txt": str(index)})
            BaseWebdatasetFactory.prepare_dataset(
                root,
                ["data.tar"],
                split_parts_ratio=[("train", 1.0)],
                shuffle_seed=None,
                workers=1,
            )
            (root / ".nv-meta/dataset.yaml").write_text(
                "__module__: megatron.energon\n__class__: TextWebdataset\nfield_map:\n  text: txt\n"
            )
            package = root / "metadata.json"
            with capture_metadata(package, recipe_identity="revision"):
                get_train_dataset(
                    root,
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
                )
            snapshot = MetadataSnapshot.load(
                package,
                sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                recipe_identity="revision",
            )
            snapshot.validate()
            (root / ".nv-meta/index.uuid").write_text("different-generation")
            with self.assertRaisesRegex(ValueError, "contents changed.*index.uuid"):
                snapshot.validate()

    def test_output_paths_and_failed_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("metadata.train", "metadata.val"):
                with capture_metadata(str(root / name), recipe_identity=name):
                    pass
                audit = json.loads((root / (name + ".audit.json")).read_text())
                self.assertEqual(
                    audit["snapshot_sha256"], hashlib.sha256((root / name).read_bytes()).hexdigest()
                )
            with self.assertRaises(FileNotFoundError):
                with capture_metadata(
                    root / "absent" / "metadata.json", recipe_identity="revision"
                ):
                    self.fail("must fail before capture starts")
            output = root / "metadata.train"
            before = output.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                with capture_metadata(output, recipe_identity="replacement"):
                    raise RuntimeError("body failed")
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(len(list(root.iterdir())), 4)

    def test_validation_scope_coverage_and_storage_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = EPath(root / "recipe.yaml")
            source.write_text("version: first\n")
            package = root / "metadata.json"
            with capture_metadata(package, recipe_identity="revision"):
                source.read_text()
            snapshot = MetadataSnapshot.load(
                package,
                sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                recipe_identity="revision",
            )
            with self.assertRaisesRegex(TypeError, "validate"):
                with use_metadata_snapshot(snapshot):
                    self.fail("unvalidated snapshot accepted")
            validated = pickle.loads(pickle.dumps(snapshot.validate()))
            with use_metadata_snapshot(validated) as replay:
                self.assertEqual(source.read_text(), "version: first\n")
                self.assertEqual(replay.hits, 1)
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    snapshot.validate()
                with self.assertRaisesRegex(ValueError, "missing contents"):
                    EPath(root / "uncaptured.yaml").read_text()
                self.assertEqual(replay.misses, 1)
            with patch("megatron.energon.epathlib.epath.NVDATASET_CACHE_DIR", EPath(root)):
                with self.assertRaisesRegex(ValueError, "NVDATASET_CACHE_DIR"):
                    snapshot.validate()
                with self.assertRaisesRegex(ValueError, "NVDATASET_CACHE_DIR"):
                    with use_metadata_snapshot(validated):
                        self.fail("different storage root accepted")

    def test_forked_loader_workers_do_not_inherit_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            test_jsonl_dataset.TestJsonlDataset.create_text_test_dataset(
                source, range(32), range(32)
            )

            def dataset():
                return get_train_dataset(
                    source,
                    worker_config=WorkerConfig(rank=0, world_size=1, num_workers=2),
                    batch_size=None,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    task_encoder=SnapshotWorkerEncoder(),
                )

            package = root / "metadata.json"
            with capture_metadata(package, recipe_identity="revision"):
                dataset()
            validated = MetadataSnapshot.load(
                package,
                sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                recipe_identity="revision",
            ).validate()
            with use_metadata_snapshot(validated):
                with get_savable_loader(dataset()) as loader:
                    iterator = iter(loader)
                    self.assertEqual(len([next(iterator) for _ in range(8)]), 8)
            self.assertIsNone(active_snapshot())


if __name__ == "__main__":
    unittest.main(verbosity=2)
