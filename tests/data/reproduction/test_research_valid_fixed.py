"""Deterministic research-validation selection for TF-Locoformer."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from src.experiments.data.errors import ReproductionError
from src.experiments.data.prepare.research_valid_fixed import (
    ALGORITHM_NAME,
    PARENT_DATASET_NAME,
    SELECTION_COUNT,
    SELECTION_SEED,
    case_id_list_sha256,
    case_id_overlap,
    clean_sha_independence,
    find_parent_snapshot,
    leakage_safe_allowed,
    noise_file_overlap,
    parent_split_case_ids,
    recording_status,
    select_case_ids,
    select_research_validation,
    selection_rank,
    speaker_independence,
    time_estimate,
    validation_budget,
)

ROOT = Path(__file__).resolve().parents[3]
LIST_JSON = ROOT / "configs" / "data" / "lists" / "tf_locoformer" / "research_valid_fixed_v1.json"


class ResearchValidFixedTest(unittest.TestCase):
    def test_rank_is_sha256_of_seed_line_and_case_id(self) -> None:
        self.assertEqual(selection_rank("abc", 20261107), hashlib.sha256(b"20261107\nabc").digest())

    def test_list_sha256_sorts_before_hashing(self) -> None:
        self.assertEqual(case_id_list_sha256(["b", "a"]), hashlib.sha256(b"a\nb\n").hexdigest())

    def test_selection_is_deterministic_and_ignores_input_order(self) -> None:
        population = [f"{index:04d}" for index in range(2000)]
        first = select_case_ids(population, seed=SELECTION_SEED, count=1024)
        second = select_case_ids(population, seed=SELECTION_SEED, count=1024)
        shuffled = list(reversed(population))
        self.assertEqual(first, second)
        self.assertEqual(select_case_ids(shuffled, seed=SELECTION_SEED, count=1024), first)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 1024)
        self.assertEqual(len(set(first)), 1024)
        self.assertTrue(set(first) <= set(population))
        ranked = sorted(population, key=lambda case_id: (selection_rank(case_id, SELECTION_SEED), case_id))
        self.assertEqual(set(first), set(ranked[:1024]))

    def test_sharded_input_matches_only_when_the_full_population_is_combined(self) -> None:
        population = [f"id-{index}" for index in range(30)]
        full = select_case_ids(population, seed=SELECTION_SEED, count=6)
        shards = [population[index::3] for index in range(3)]
        merged = [item for shard in reversed(shards) for item in shard]
        self.assertEqual(select_case_ids(merged, seed=SELECTION_SEED, count=6), full)
        per_shard = []
        for shard in shards:
            per_shard.extend(select_case_ids(shard, seed=SELECTION_SEED, count=2))
        self.assertNotEqual(sorted(set(per_shard)), full)

    def test_selector_does_not_mutate_input_or_parent_dataset_name(self) -> None:
        self.assertEqual(PARENT_DATASET_NAME, "tf_locoformer.dns.accept_recipe")
        population = ["c", PARENT_DATASET_NAME, "a", "b"]
        original = list(population)
        select_case_ids(population, seed=SELECTION_SEED, count=2)
        self.assertEqual(population, original)
        self.assertEqual(PARENT_DATASET_NAME, "tf_locoformer.dns.accept_recipe")

    def test_module_does_not_call_random(self) -> None:
        text = (ROOT / "src" / "experiments" / "data" / "prepare" / "research_valid_fixed.py").read_text(encoding="utf-8")
        self.assertNotIn("import random", text)
        self.assertNotIn("random.random", text)
        self.assertNotIn("default_rng", text)
        self.assertNotIn("numpy.random", text)

    def test_duplicates_and_oversized_count_are_rejected(self) -> None:
        with self.assertRaises(ReproductionError):
            select_case_ids(["a", "a", "b"], seed=SELECTION_SEED, count=1)
        with self.assertRaises(ReproductionError):
            select_case_ids(["a", "b"], seed=SELECTION_SEED, count=3)

    def test_fake_train_and_test_do_not_overlap_the_selection(self) -> None:
        valid = [f"v{index:04d}" for index in range(3000)]
        train = [f"t{index:04d}" for index in range(10)]
        test = [f"e{index:04d}" for index in range(10)]
        chosen = select_research_validation(valid, train, test, seed=SELECTION_SEED, count=1024)
        self.assertEqual(chosen["count"], 1024)
        self.assertEqual(len(set(chosen["case_ids"])), 1024)
        self.assertTrue(set(chosen["case_ids"]) <= set(valid))
        self.assertEqual(chosen["case_id_overlap"]["train"]["status"], "PASS")
        self.assertEqual(chosen["case_id_overlap"]["test"]["status"], "PASS")
        self.assertFalse(chosen["reduced_because_leak"])
        overlap = case_id_overlap(chosen["case_ids"], train, test)
        self.assertEqual(overlap["train"]["count"], 0)
        self.assertEqual(overlap["test"]["count"], 0)

    def test_train_leak_can_force_a_smaller_set(self) -> None:
        valid = [f"v{index:04d}" for index in range(10)]
        train = valid[:8]
        chosen = select_research_validation(valid, train, [], seed=SELECTION_SEED, count=4)
        self.assertEqual(chosen["count"], 2)
        self.assertTrue(chosen["reduced_because_leak"])
        self.assertEqual(chosen["case_id_overlap"]["train"]["status"], "PASS")
        self.assertTrue(set(chosen["case_ids"]).isdisjoint(train))

    def test_absent_speaker_is_unknown_not_pass(self) -> None:
        report = speaker_independence([None, None], [None])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertNotEqual(report["status"], "PASS")
        present = speaker_independence(["s1"], ["s2"])
        self.assertEqual(present["status"], "PASS")

    def test_absent_clean_sha_is_unknown_not_pass(self) -> None:
        report = clean_sha_independence([None], ["abc"])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertNotEqual(report["status"], "PASS")
        self.assertEqual(clean_sha_independence(["aa"], ["bb"])["status"], "PASS")
        self.assertEqual(clean_sha_independence(["aa"], ["aa"])["status"], "OVERLAP")

    def test_absent_recording_id_is_unknown_not_pass(self) -> None:
        report = recording_status(["name", "snr_db", "start"])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertNotEqual(report["status"], "PASS")

    def test_noise_overlap_is_counted_and_blocks_leakage_safe(self) -> None:
        overlap = noise_file_overlap(shared_names=3, cases_sharing=2)
        self.assertEqual(overlap["status"], "OVERLAP")
        self.assertEqual(overlap["shared_noise_names"], 3)
        checks = {
            "case_id_train": {"status": "PASS"},
            "case_id_test": {"status": "PASS"},
            "clean_sha256_in_case_document": {"status": "UNKNOWN"},
            "speaker": {"status": "UNKNOWN"},
            "recording": {"status": "UNKNOWN"},
            "noise_file_name": overlap,
        }
        self.assertFalse(leakage_safe_allowed(checks))
        self.assertEqual(noise_file_overlap(shared_names=0, cases_sharing=0)["status"], "PASS")

    def test_published_validation_budget(self) -> None:
        full = validation_budget(36000, epochs=150)
        subset = validation_budget(1024, epochs=150)
        self.assertEqual(full["chunks_per_rank"], 126000)
        self.assertEqual(full["chunks"], 504000)
        self.assertEqual(full["one_pass_hours"], "5.607")
        self.assertEqual(full["all_epochs_hours"], "841.05")
        self.assertEqual(subset["utterances"], 1024)
        self.assertEqual(subset["chunks"], 14336)
        self.assertEqual(subset["chunks_per_rank"], 3584)
        self.assertEqual(subset["one_pass_seconds"], "574.1568")
        self.assertEqual(subset["one_pass_hours"], "0.159488")
        self.assertEqual(subset["all_epochs_hours"], "23.9232")
        estimate = time_estimate()
        self.assertEqual(estimate["label"], "extrapolated")
        self.assertTrue(estimate["not_remeasured"])
        self.assertEqual(estimate["training_hours"], "384.4")
        self.assertEqual(estimate["checkpoint"]["write_seconds"], "343.5")
        self.assertEqual(estimate["checkpoint"]["kept_files_upper_bound_if_roles_do_not_overlap"], 10)
        self.assertEqual(estimate["total_hours_rounded_4dp"], "408.4186")

    def test_parent_validation_subset_when_snapshot_is_present(self) -> None:
        root = ROOT / "data" / "db" / "snapshots"
        if not root.is_dir():
            self.skipTest("parent snapshots are not on this machine")
        snapshot = find_parent_snapshot(root)
        parent = parent_split_case_ids(snapshot)
        self.assertEqual(len(parent["valid"]), 36000)
        self.assertEqual(len(parent["train"]), 324000)
        self.assertEqual(len(parent["test"]), 150)
        self.assertEqual(parent["dataset_id"], "5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe")
        chosen = select_research_validation(parent["valid"], parent["train"], parent["test"])
        self.assertEqual(chosen["algorithm"], ALGORITHM_NAME)
        self.assertEqual(chosen["seed"], SELECTION_SEED)
        self.assertEqual(chosen["count"], SELECTION_COUNT)
        self.assertEqual(len(set(chosen["case_ids"])), SELECTION_COUNT)
        self.assertTrue(set(chosen["case_ids"]) <= set(parent["valid"]))
        self.assertEqual(chosen["case_id_overlap"]["train"]["status"], "PASS")
        self.assertEqual(chosen["case_id_overlap"]["test"]["status"], "PASS")
        self.assertEqual(PARENT_DATASET_NAME, "tf_locoformer.dns.accept_recipe")
        if not LIST_JSON.is_file():
            return
        payload = json.loads(LIST_JSON.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_id"], "tf_locoformer.dns.accept_recipe.research_valid_fixed@1")
        self.assertEqual(payload["case_ids"], chosen["case_ids"])
        self.assertEqual(payload["case_id_list_sha256"], chosen["case_id_list_sha256"])
        self.assertEqual(payload["registration_status"], "BLOCKED")
        self.assertIsNone(payload["snapshot_hash"])
        self.assertEqual(payload["leak"]["checks"]["speaker"]["status"], "UNKNOWN")
        self.assertNotEqual(payload["leak"]["checks"]["speaker"]["status"], "PASS")
        self.assertFalse(payload["leakage_safe"])
        self.assertFalse(payload["selection_uses_test"])
        self.assertNotIn("cv_synthetic_small", payload["claim_label"])
        self.assertNotIn("official_reproduction", payload["claim_label"])


if __name__ == "__main__":
    unittest.main()
