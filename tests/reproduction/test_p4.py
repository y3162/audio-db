"""P4 provenance propagation. Synthetic rows only, except the real-snapshot smoke."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.datasets.load import DatasetBundle
from src.evaluation.datasets.synthetic import make_synthetic_samples
from src.evaluation.pipeline import execute
from src.evaluation.provenance import UNKNOWN, is_confirmed_independent, judge_independence
from src.reproduction.db.registry import initialize_database, register_staging, register_training_run
from src.reproduction.db.staging import write_graph
from src.reproduction.errors import ReproductionError
from src.reproduction.paths import DEFAULT_RESOURCE_DB, DEFAULT_SNAPSHOTS, ROOT
from src.reproduction.pipeline.training_data import open_training_data
from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint, save_atomic
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from src.training.provenance import (
    assert_checkpoint_provenance,
    build_provenance,
    publish_provenance,
)
from tests.reproduction.support import fixed_pair_graph
from tests.training.mock_task import LinearPairTask
from tests.training.support import cpu_process, paired_dataset, tiny_config

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
DATASETS = (
    "mp_senet.vb28.official@1",
    "mp_senet.vb28.safe@1",
    "mossformergan.vb28.official@1",
    "mossformergan.vb28.safe@1",
)


class _Identity:
    def __init__(self, identity: dict):
        self._identity = identity

    def identity(self):
        return dict(self._identity)


class ProvenanceCheckpointTests(unittest.TestCase):
    def test_run_id_follows_the_dataset_and_not_the_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(root / "one", max_steps=1, seed=7, loss="mock")
            left = paired_dataset(root / "wav", 2, frames=32, segment_size=16, seed=7, name="left")
            right = paired_dataset(root / "wav", 2, frames=32, segment_size=16, seed=7, rewrite=False, name="left")
            other = paired_dataset(root / "other", 2, frames=32, segment_size=16, seed=7, name="right")
            process = cpu_process()
            first = build_provenance(left, config, process)
            second = build_provenance(right, config, process)
            changed = build_provenance(other, config, process)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["pipeline"], "legacy")
        self.assertNotEqual(first["run_id"], changed["run_id"])
        self.assertIsNone(first["selection_uses_test"])
        self.assertIn("legacy", " ".join(first["limitations"]))

    def test_rank_other_than_zero_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = build_provenance(
                paired_dataset(root / "wav", 2, frames=32, segment_size=16),
                tiny_config(root / "ckpt", max_steps=1, seed=7, loss="mock"),
                cpu_process(),
            )
            with self.assertRaisesRegex(TrainingError, "rank 0"):
                publish_provenance(root / "ckpt", document, rank=1)
            self.assertFalse((root / "ckpt" / "data_provenance.json").exists())

    def test_same_run_is_kept_and_a_different_dataset_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = paired_dataset(root / "wav", 2, frames=32, segment_size=16, name="left")
            config = tiny_config(root / "ckpt", max_steps=1, seed=7, loss="mock")
            document = build_provenance(dataset, config, cpu_process())
            raw = publish_provenance(root / "ckpt", document, rank=0)
            again = publish_provenance(root / "ckpt", document, rank=0)
            self.assertEqual(raw, again)
            other = build_provenance(
                paired_dataset(root / "other", 2, frames=32, segment_size=16, name="right"),
                config,
                cpu_process(),
            )
            with self.assertRaisesRegex(TrainingError, "does not match this run"):
                publish_provenance(root / "ckpt", other, rank=0)
            self.assertEqual((root / "ckpt" / "data_provenance.json").read_bytes(), raw)

    def test_trainer_writes_matching_checkpoint_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, _trainer = _fit(root, steps=1)
            checkpoint = root / "ckpt" / "step_00000001.pt"
            payload = load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(payload["format"], "audio-db-training-v1")
            provenance_path = root / "ckpt" / "data_provenance.json"
            sidecar = checkpoint.with_suffix(".data_provenance.json")
            self.assertEqual(provenance_path.read_bytes(), sidecar.read_bytes())
            self.assertEqual(hashlib.sha256(provenance_path.read_bytes()).hexdigest(), payload["provenance_sha256"])
            document = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(document["run_id"], payload["run_id"])
            self.assertEqual(document["pipeline"], "legacy")
            self.assertTrue((root / "ckpt" / "generator.data_provenance.json").is_file())
            assert_checkpoint_provenance(payload, checkpoint, paired_dataset(root / "wav", 4, frames=80, segment_size=40, rewrite=False))
            del task

    def test_corrupt_or_missing_provenance_is_not_a_valid_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fit(root, steps=1)
            checkpoint = root / "ckpt" / "step_00000001.pt"
            payload = load_checkpoint(checkpoint, torch.device("cpu"))
            dataset = paired_dataset(root / "wav", 4, frames=80, segment_size=40, rewrite=False)
            provenance = root / "ckpt" / "data_provenance.json"
            provenance.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "unreadable"):
                assert_checkpoint_provenance(payload, checkpoint, dataset)
            provenance.unlink()
            checkpoint.with_suffix(".data_provenance.json").unlink()
            with self.assertRaisesRegex(TrainingError, "missing"):
                assert_checkpoint_provenance(payload, checkpoint, dataset)

    def test_legacy_checkpoint_without_provenance_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            payload = {
                "format": "audio-db-training-v1",
                "global_step": 1,
                "epoch": 1,
                "micro_batches_consumed": 1,
                "task": {},
                "rng": {},
                "dataset": {"name": "old"},
                "config": {},
                "batch_plan": {},
            }
            save_atomic(path, payload, verify_training=True)
            loaded = load_checkpoint(path, torch.device("cpu"))
            self.assertNotIn("provenance_sha256", loaded)
            assert_checkpoint_provenance(loaded, path, _Identity({"name": "old"}))

    def test_resume_rejects_a_different_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fit(root, steps=1)
            checkpoint = root / "ckpt" / "step_00000001.pt"
            payload = load_checkpoint(checkpoint, torch.device("cpu"))
            with self.assertRaisesRegex(TrainingError, "does not match this dataset"):
                assert_checkpoint_provenance(payload, checkpoint, _Identity({"name": "different", "case_ids_sha256": "0" * 64}))


class RunRegistrationTests(unittest.TestCase):
    def test_register_distinguishes_resume_from_a_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "reproduction.duckdb"
            initialize_database(database)
            graph = fixed_pair_graph(root / "corpora", safe=False)
            staging = root / "staging"
            write_graph(staging, graph)
            register_staging(staging, database)
            dataset = graph.datasets[0]
            provenance, state = _run_record(dataset, snapshot="ab" * 32, epoch=1, step="step_00000001.pt")
            first = register_training_run(
                database,
                provenance,
                state,
                checkpoint_path="step_00000001.pt",
                checkpoint_sha256="1" * 64,
            )
            second = register_training_run(
                database,
                provenance,
                state,
                checkpoint_path="step_00000001.pt",
                checkpoint_sha256="1" * 64,
            )
            self.assertEqual(first, "inserted")
            self.assertEqual(second, "unchanged")
            continued_state = dict(state)
            continued_state["epoch"] = 2
            continued_state["plan_hash"] = "9" * 64
            continued = register_training_run(
                database,
                provenance,
                continued_state,
                checkpoint_path="step_00000002.pt",
                checkpoint_sha256="2" * 64,
            )
            self.assertEqual(continued, "updated")
            changed = dict(provenance)
            changed["snapshot_hash"] = "cd" * 32
            changed_state = dict(continued_state)
            changed_state["snapshot_hash"] = changed["snapshot_hash"]
            with self.assertRaisesRegex(ReproductionError, "snapshot_hash"):
                register_training_run(
                    database,
                    changed,
                    changed_state,
                    checkpoint_path="step_00000002.pt",
                    checkpoint_sha256="2" * 64,
                )
            fresh = dict(provenance)
            fresh["run_id"] = "4" * 64
            inserted = register_training_run(
                database,
                fresh,
                state,
                checkpoint_path="other/step_00000001.pt",
                checkpoint_sha256="3" * 64,
            )
            self.assertEqual(inserted, "inserted")
            import duckdb

            connection = duckdb.connect(str(database), read_only=True)
            try:
                rows = connection.execute(
                    "SELECT run_id, dataset_id, snapshot_hash, selection_uses_test, data_state_json::VARCHAR FROM repro.runs WHERE run_id = ?",
                    ["7" * 64],
                ).fetchall()
                count = connection.execute("SELECT count(*) FROM repro.runs").fetchone()[0]
            finally:
                connection.close()
        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 1)
        stored = json.loads(rows[0][4])
        self.assertEqual(stored["claim_label"], dataset["claim_label"])
        self.assertEqual(stored["profile"], dataset["profile"])
        self.assertEqual(stored["checkpoint_path"], "step_00000002.pt")
        self.assertEqual(stored["epoch"], 2)
        self.assertIs(rows[0][3], True)


class EvaluationPropagationTests(unittest.TestCase):
    def test_three_judgments_are_recorded_on_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fit(root / "overlap", steps=1, count=4)
            _fit(root / "clear", steps=1, count=4)
            _fit(root / "missing", steps=1, count=4)
            overlap = _publish_case(
                root / "overlap" / "ckpt",
                selection=SHA_A,
                train=SHA_B,
                test=SHA_C,
                selection_uses_test=True,
                profile="official_faithful",
                claim="official_reproduction",
            )
            clear = _publish_case(
                root / "clear" / "ckpt",
                selection=SHA_B,
                train=SHA_A,
                test=SHA_C,
            )
            missing_checkpoint = root / "missing" / "ckpt" / "generator.pt"
            results = {
                "overlap": _evaluate(root / "out-overlap", overlap, SHA_A, "selected_test"),
                "clear": _evaluate(root / "out-clear", clear, "f" * 64, "other_set"),
                "missing": _evaluate(root / "out-missing", missing_checkpoint, "f" * 64, "other_set"),
            }
        self.assertIs(results["overlap"]["independent"], False)
        self.assertEqual(results["overlap"]["checkpoint_selection_overlap"], ["selected_test"])
        self.assertIs(results["overlap"]["data_independence"]["selection_uses_test"], True)
        self.assertIs(results["clear"]["independent"], True)
        self.assertEqual(results["clear"]["checkpoint_selection_overlap"], [])
        self.assertTrue(is_confirmed_independent(results["clear"]["independent"]))
        self.assertEqual(results["missing"]["independent"], UNKNOWN)
        self.assertIsNone(results["missing"]["checkpoint_selection_overlap"])
        self.assertFalse(is_confirmed_independent(results["missing"]["independent"]))
        self.assertEqual(results["missing"]["summary_independent"], UNKNOWN)


class RealSnapshotSmokeTests(unittest.TestCase):
    def test_four_datasets_round_trip_provenance(self) -> None:
        resource = DEFAULT_RESOURCE_DB
        reproduction = ROOT / "data" / "db" / "reproduction.duckdb"
        before = (_stat(resource), _stat(reproduction))
        import duckdb

        connection = duckdb.connect(str(resource), read_only=True)
        try:
            schemas = {row[0] for row in connection.execute("SELECT schema_name FROM information_schema.schemata").fetchall()}
        finally:
            connection.close()
        self.assertNotIn("eval", schemas)
        process = cpu_process()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for spec in DATASETS:
                config = tiny_config(root / spec, max_steps=1, seed=1, loss="mock")
                data = open_training_data(
                    dataset=spec,
                    allow_test_selection=spec.endswith("official@1"),
                    seed=1,
                    global_batch=2,
                    micro_batch=1,
                    accumulation=1,
                    world_size=1,
                    snapshot_root=DEFAULT_SNAPSHOTS,
                    load_workers=1,
                )
                try:
                    document = build_provenance(data, config, process)
                finally:
                    data.close()
                self.assertEqual(document["dataset_name"] + f"@{document['dataset_version']}", spec)
                self.assertEqual(document["snapshot_hash"], data.snapshot_hash)
                raw = publish_provenance(root / spec, document, rank=0)
                loaded = json.loads((root / spec / "data_provenance.json").read_bytes())
                self.assertEqual(canonical_bytes_equal(raw, loaded), True)
                judged = judge_independence(loaded, None)
                reports.append(
                    {
                        "dataset": spec,
                        "profile": loaded["profile"],
                        "claim_label": loaded["claim_label"],
                        "selection_uses_test": loaded["selection_uses_test"],
                        "split_independent": judged["split_independent"],
                        "independent": judged["independent"],
                        "comparison": judged["comparison"],
                        "lineage_counts": loaded["lineage_counts"],
                        "run_id": loaded["run_id"],
                        "snapshot_hash": loaded["snapshot_hash"],
                    }
                )
        self.assertEqual(_stat(resource), before[0])
        self.assertEqual(_stat(reproduction), before[1])
        by_name = {item["dataset"]: item for item in reports}
        for name in ("mp_senet.vb28.official@1", "mossformergan.vb28.official@1"):
            self.assertIs(by_name[name]["selection_uses_test"], True)
            self.assertEqual(by_name[name]["profile"], "official_faithful")
            self.assertEqual(by_name[name]["claim_label"], "official_reproduction")
            self.assertIs(by_name[name]["split_independent"], False)
            self.assertEqual(by_name[name]["independent"], UNKNOWN)
            self.assertEqual(by_name[name]["comparison"], "NOT_TESTED")
            self.assertEqual(by_name[name]["lineage_counts"]["unknown"], 0)
            self.assertGreater(by_name[name]["lineage_counts"]["derived"], 0)
        for name in ("mp_senet.vb28.safe@1", "mossformergan.vb28.safe@1"):
            self.assertIs(by_name[name]["selection_uses_test"], False)
            self.assertEqual(by_name[name]["profile"], "leakage_safe")
            self.assertEqual(by_name[name]["claim_label"], "leakage_safe_comparison")
            self.assertIs(by_name[name]["split_independent"], True)
            self.assertEqual(by_name[name]["independent"], UNKNOWN)
            self.assertEqual(by_name[name]["comparison"], "NOT_TESTED")
            self.assertEqual(by_name[name]["lineage_counts"]["unknown"], 0)
        self.reports = reports


def _fit(root: Path, *, steps: int, count: int = 4):
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config(
        root / "ckpt",
        max_steps=steps,
        seed=7,
        global_batch_size=4,
        micro_batch_size_per_gpu=4,
        loss="mock",
        checkpoint_interval=1,
    )
    dataset = paired_dataset(root / "wav", count, frames=80, segment_size=40, seed=7)
    task = LinearPairTask(config, torch.device("cpu"))
    trainer = Trainer(
        task,
        plan_batches(4, 4, 1),
        cpu_process(),
        max_steps=steps,
        seed=7,
        checkpoint_dir=root / "ckpt",
        checkpoint_interval=1,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset)
    return task, trainer


def _run_record(dataset: dict, *, snapshot: str, epoch: int, step: str) -> tuple[dict, dict]:
    state = {
        "pipeline": "reproduction",
        "dataset_id": dataset["dataset_id"],
        "snapshot_hash": snapshot,
        "profile": dataset["profile"],
        "selection_uses_test": dataset["selection_uses_test"],
        "world_size": 1,
        "epoch": epoch,
        "update_in_epoch": 0,
        "plan_hash": "8" * 64,
        "seed": 1,
    }
    provenance = {
        "pipeline": "reproduction",
        "run_id": "7" * 64,
        "dataset_id": dataset["dataset_id"],
        "snapshot_hash": snapshot,
        "config_hash": "6" * 64,
        "git_commit": "synthetic",
        "ddp_mode": "single",
        "profile": dataset["profile"],
        "claim_label": dataset["claim_label"],
        "selection_uses_test": dataset["selection_uses_test"],
    }
    del step
    return provenance, state


def _publish_case(
    directory: Path,
    *,
    selection: str,
    train: str,
    test: str,
    selection_uses_test: bool = False,
    profile: str = "leakage_safe",
    claim: str = "leakage_safe_comparison",
) -> Path:
    materials = [
        {"role": "train", "slot": "train.cases", "case_set_id": "train-set", "asset_id": "corpora/train.wav", "sha256": train, "source_sha256": None, "lineage": "source"},
        {"role": "selection", "slot": "valid.cases", "case_set_id": "valid-set", "asset_id": "corpora/valid.wav", "sha256": selection, "source_sha256": None, "lineage": "source"},
        {"role": "test", "slot": "test.cases", "case_set_id": "test-set", "asset_id": "corpora/test.wav", "sha256": test, "source_sha256": None, "lineage": "source"},
    ]
    document = {
        "schema_version": 1,
        "format": "audio-db-data-provenance-v1",
        "pipeline": "reproduction",
        "run_id": "1" * 64,
        "model_name": "mp_senet",
        "dataset_id": "2" * 64,
        "dataset_name": "synthetic",
        "dataset_version": 1,
        "snapshot_hash": "3" * 64,
        "protocol_id": "protocol",
        "profile": profile,
        "claim_label": claim,
        "selection_uses_test": selection_uses_test,
        "git_commit": "synthetic",
        "config_hash": "4" * 64,
        "ddp_mode": "single",
        "world_size": 1,
        "seed": 7,
        "sets": [],
        "materials": materials,
        "lineage_counts": {"source": 3, "derived": 0, "unknown": 0},
        "limitations": [],
        "data_sha256": "5" * 64,
        "dataset_identity": {"name": "replaced"},
    }
    from src.evaluation.provenance import canonical_bytes

    raw = canonical_bytes(document)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data_provenance.json").write_bytes(raw)
    checkpoint = directory / "generator.pt"
    if not checkpoint.is_file():
        checkpoint.write_bytes(b"not-opened-by-the-mock-factory")
    checkpoint.with_suffix(".data_provenance.json").write_bytes(raw)
    return checkpoint


def _evaluate(root: Path, checkpoint: Path, sha: str, set_name: str) -> dict:
    sample = make_synthetic_samples(1, 16000, 0.05, 0)[0]
    sample.dataset = set_name
    sample.metadata["material_identity"] = {
        "set_name": set_name,
        "sha256": sha,
        "source_sha256": None,
        "lineage": "source",
    }
    from tests.evaluation.support import experiment_config

    config = experiment_config(
        root,
        run_id="prov",
        models=[{"name": "mp_senet", "checkpoint": str(checkpoint)}],
        asr={"enabled": False, "name": "mock"},
        metrics={"wer": False, "pesq": False, "stoi": False, "estoi": False, "si_sdr": False},
        mixing={"enabled": False},
        dataset={"source": "synthetic", "count": 1, "sample_rate": 16000, "seconds": 0.05, "seed": 0},
        experiment={"name": "provenance", "seed": 0, "limit": 1},
    )

    def factory(model, device):
        del model, device

        class _Copy:
            def enhance(self, waveform, sample_rate):
                del sample_rate
                return waveform.clone()

        return _Copy()

    result = execute(config, enhancer_factory=factory, bundle=DatasetBundle(samples=[sample]))
    record = result["records"][0]
    return {
        "independent": record["independent"],
        "checkpoint_selection_overlap": record["checkpoint_selection_overlap"],
        "data_independence": record["data_independence"],
        "summary_independent": result["summary"]["data_independence"]["independent"],
    }


def _stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def canonical_bytes_equal(raw: bytes, loaded: dict) -> bool:
    from src.evaluation.provenance import canonical_bytes

    return canonical_bytes(loaded) == raw


if __name__ == "__main__":
    unittest.main()
