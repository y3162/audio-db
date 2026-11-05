"""P1 acceptance tests. All audio and databases here are synthetic."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.reproduction.db.graph import Graph
from src.reproduction.db.registry import initialize_database, read_graph, rebuild_database, register_staging, table_hashes
from src.reproduction.db.schema import create_schema
from src.reproduction.db.staging import load_graph, write_graph
from src.reproduction.errors import ReproductionError
from src.reproduction.paths import DEFAULT_RESOURCE_DB, ROOT
from tests.reproduction.support import fixed_pair_graph, generated_graph

RESOURCE = DEFAULT_RESOURCE_DB


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, _sha256(path)


_RESOURCE_BEFORE = _fingerprint(RESOURCE)
_DB_DIR_BEFORE = sorted(path.name for path in RESOURCE.parent.iterdir())


def tearDownModule() -> None:
    after = _fingerprint(RESOURCE)
    if after != _RESOURCE_BEFORE:
        raise AssertionError("data/db/resource.duckdb changed during reproduction tests")
    names = sorted(path.name for path in RESOURCE.parent.iterdir())
    if names != _DB_DIR_BEFORE:
        raise AssertionError(f"data/db directory changed: {names}")


class ReproductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.corpora = self.root / "corpora"
        self.generated = self.root / "generated"
        self.staging = self.root / "staging"
        self.db = self.root / "reproduction.duckdb"
        self.corpora.mkdir()
        initialize_database(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_register_order_and_rebuild_match(self) -> None:
        official = fixed_pair_graph(self.corpora, safe=False)
        safe = fixed_pair_graph(self.corpora, safe=True)
        official_dir = self.staging / "official"
        safe_dir = self.staging / "safe"
        both = self.staging / "both"
        write_graph(official_dir, official)
        write_graph(safe_dir, safe)
        write_graph(both, official)
        write_graph(both, safe)
        register_staging(official_dir, self.db)
        register_staging(safe_dir, self.db)
        forward = table_hashes(self.db)

        reverse_db = self.root / "reverse.duckdb"
        initialize_database(reverse_db)
        register_staging(safe_dir, reverse_db)
        register_staging(official_dir, reverse_db)
        self.assertEqual(forward, table_hashes(reverse_db))

        again = register_staging(both, self.db)
        self.assertTrue(all(count == 0 for count in again.values()))
        self.assertEqual(forward, table_hashes(self.db))

        rebuilt = self.root / "rebuilt.duckdb"
        rebuild_database(both, rebuilt, reference=self.db)
        self.assertEqual(forward, table_hashes(rebuilt))
        with self.assertRaises(ReproductionError):
            rebuild_database(both, rebuilt, reference=self.db)

    def test_same_id_different_content_rolls_back(self) -> None:
        graph = fixed_pair_graph(self.corpora, safe=False)
        write_graph(self.staging, graph)
        register_staging(self.staging, self.db)
        before = table_hashes(self.db)
        changed = fixed_pair_graph(self.corpora, safe=False)
        target = changed.assets[0]
        target["frames"] = 3200
        extra = dict(changed.assets[-1])
        extra["rel_path"] = "official/extra.wav"
        extra["asset_id"] = "corpora/official/extra.wav"
        extra["utterance_key"] = "extra"
        changed.assets.append(extra)
        other = self.root / "other-staging"
        write_graph(other, changed)
        with self.assertRaises(ReproductionError):
            register_staging(other, self.db)
        self.assertEqual(before, table_hashes(self.db))
        stored = read_graph(self.db)
        self.assertNotIn(extra["asset_id"], {row["asset_id"] for row in stored.assets})

    def test_claim_selection_and_sql_checks(self) -> None:
        graph = fixed_pair_graph(self.corpora, safe=False)
        graph.datasets[0]["selection_uses_test"] = False
        write_graph(self.staging, graph)
        with self.assertRaises(ReproductionError) as caught:
            register_staging(self.staging, self.db)
        self.assertIn("selection_uses_test", str(caught.exception))
        self.assertEqual([], read_graph(self.db).datasets)

        safe = fixed_pair_graph(self.corpora, safe=True)
        safe.datasets[0]["claim_label"] = "official_reproduction"
        with self.assertRaises(ReproductionError) as claim:
            write_and_register(self.root / "bad-claim", safe, self.db)
        self.assertIn("leakage_safe", str(claim.exception))

        import duckdb

        connection = duckdb.connect(str(self.db))
        try:
            protocol = graph.protocols[0]
            _insert_protocol(connection, protocol)
            with self.assertRaises(Exception):
                connection.execute(
                    """
                    INSERT INTO repro.datasets VALUES (
                        'dataset-bad', 'bad', 1, ?, 'leakage_safe', 'official',
                        'leakage_safe_comparison', 'ready', '[]'::JSON, true, NULL, '[]'::JSON
                    )
                    """,
                    [protocol["protocol_id"]],
                )
            with self.assertRaises(Exception):
                connection.execute(
                    """
                    INSERT INTO repro.assets VALUES (
                        'generated/missing.wav', 'generated', 'missing.wav',
                        NULL, NULL, NULL, NULL, 16000, 1600, 1, NULL, 4,
                        'aa', 'sha256', NULL, NULL, NULL, NULL, '{}'::JSON
                    )
                    """
                )
        finally:
            connection.close()

    def test_incomplete_generated_is_rejected(self) -> None:
        graph = generated_graph(self.root)
        graph.verified = {}
        write_graph(self.staging, graph)
        before = _sha256(self.db)
        with self.assertRaises(ReproductionError) as caught:
            register_staging(self.staging, self.db)
        self.assertIn("verified", str(caught.exception))
        self.assertEqual(before, _sha256(self.db))
        self.assertEqual([], read_graph(self.db).assets)

    def test_snapshot_roundtrip_and_reader_without_duckdb(self) -> None:
        graph = fixed_pair_graph(self.corpora, safe=False)
        write_graph(self.staging, graph)
        register_staging(self.staging, self.db)
        from src.reproduction.db.snapshot import export_snapshot
        from src.reproduction.snapshot_reader import load_snapshot

        snapshots = self.root / "snapshots"
        digest = export_snapshot(
            self.db,
            "synthetic.official@1",
            snapshots,
            corpora=self.corpora,
            generated=self.generated,
            derived=self.root / "derived",
        )
        loaded = load_snapshot(snapshots / _dataset_id(self.db, "synthetic.official"))
        self.assertEqual(digest, loaded.snapshot_hash)
        self.assertTrue(loaded.assets)
        self.assertTrue(loaded.docs)
        manifest = snapshots / loaded.directory.name / "manifest.json"
        mtime = manifest.stat().st_mtime_ns
        second = export_snapshot(
            self.db,
            loaded.dataset["dataset_id"],
            snapshots,
            corpora=self.corpora,
            generated=self.generated,
            derived=self.root / "derived",
        )
        self.assertEqual(digest, second)
        self.assertEqual(mtime, manifest.stat().st_mtime_ns)

        safe = fixed_pair_graph(self.corpora, safe=True)
        write_graph(self.staging / "safe", safe)
        register_staging(self.staging / "safe", self.db)
        with self.assertRaises(ReproductionError) as caught:
            export_snapshot(
                self.db,
                "synthetic.safe@1",
                snapshots,
                corpora=self.corpora,
                generated=self.generated,
                derived=self.root / "derived",
            )
        self.assertIn("UNKNOWN", str(caught.exception))
        allowed = export_snapshot(
            self.db,
            "synthetic.safe@1",
            snapshots,
            allow_unknown_leak_check=True,
            corpora=self.corpora,
            generated=self.generated,
            derived=self.root / "derived",
        )
        safe_manifest = json.loads((snapshots / _dataset_id(self.db, "synthetic.safe") / "manifest.json").read_text())
        self.assertTrue(safe_manifest["provenance"]["allow_unknown_leak_check"])
        self.assertNotIn("allow_unknown_leak_check", json.dumps(load_snapshot(snapshots / _dataset_id(self.db, "synthetic.safe")).dataset))
        self.assertTrue(allowed)

        with self.assertRaises(ReproductionError):
            export_snapshot(
                self.db,
                "synthetic.blocked@1",
                snapshots,
                corpora=self.corpora,
                generated=self.generated,
                derived=self.root / "derived",
            )

        reader = ROOT / ".venvs" / "deepfilternet" / "bin" / "python"
        self.assertTrue(reader.is_file(), "deepfilternet environment is required for the snapshot reader check")
        script = (
            "import sys\n"
            "from src.reproduction.snapshot_reader import load_snapshot\n"
            "loaded = load_snapshot(sys.argv[1])\n"
            "print(loaded.snapshot_hash)\n"
            "print('duckdb' in sys.modules)\n"
        )
        result = subprocess.run(
            [str(reader), "-c", script, str(snapshots / loaded.directory.name)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[-2], digest)
        self.assertEqual(lines[-1], "False")

    def test_generated_snapshot_uses_generated_json(self) -> None:
        graph = generated_graph(self.root)
        write_graph(self.staging, graph)
        register_staging(self.staging, self.db)
        from src.reproduction.db.snapshot import export_snapshot
        from src.reproduction.snapshot_reader import load_snapshot

        snapshots = self.root / "snapshots"
        export_snapshot(
            self.db,
            "synthetic.generated@1",
            snapshots,
            corpora=self.corpora,
            generated=self.generated,
            derived=self.root / "derived",
        )
        loaded = load_snapshot(snapshots / _dataset_id(self.db, "synthetic.generated"))
        self.assertEqual(loaded.assets, [])
        self.assertGreaterEqual(len(loaded.generated), 6)
        self.assertEqual(loaded.docs, [])

    def test_hash_mismatch_is_stored_and_does_not_change_sha(self) -> None:
        graph = fixed_pair_graph(self.corpora, safe=False)
        write_graph(self.staging, graph)
        register_staging(self.staging, self.db)
        target = next(row for row in read_graph(self.db).assets if row["integrity"] == "sha256")
        path = self.corpora / target["rel_path"]
        path.write_bytes(b"x" * path.stat().st_size)
        from src.reproduction.db.verify import verify_files

        ok = verify_files(
            self.db,
            staging=self.staging,
            dataset="synthetic.official@1",
            check_hash=True,
            corpora=self.corpora,
            generated=self.generated,
            derived=self.root / "derived",
        )
        self.assertFalse(ok)
        stored = next(row for row in read_graph(self.db).assets if row["asset_id"] == target["asset_id"])
        self.assertEqual(stored["sha256"], target["sha256"])
        audits = [row for row in read_graph(self.db).audits if row["audit_kind"] == "hash_verify"]
        self.assertTrue(any(row["status"] == "FAIL" for row in audits))

    def test_missing_file_and_hash_backfill(self) -> None:
        graph = fixed_pair_graph(self.corpora, safe=False)
        write_graph(self.staging, graph)
        register_staging(self.staging, self.db)
        orphan_id = graph.orphan_id
        self.assertIsNone(next(row for row in read_graph(self.db).assets if row["asset_id"] == orphan_id)["sha256"])
        payload = (self.corpora / "official" / "orphan.wav").read_bytes()
        update = {
            "asset_id": orphan_id,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        from src.reproduction.ids import audit_id_for

        dataset_id = _dataset_id(self.db, "synthetic.official")
        report = {"status": "PASS", "hash": True, "checks": [{"asset_id": orphan_id, "reason": "filled"}]}
        audit = {
            "audit_id": audit_id_for(
                subject_kind="dataset",
                subject_id=dataset_id,
                audit_kind="hash_verify",
                report=report,
            ),
            "subject_kind": "dataset",
            "subject_id": dataset_id,
            "audit_kind": "hash_verify",
            "status": "PASS",
            "report_json": report,
        }
        write_graph(self.staging, Graph(audits=[audit], hash_updates=[update]))
        register_staging(self.staging, self.db)
        filled = next(row for row in read_graph(self.db).assets if row["asset_id"] == orphan_id)
        self.assertEqual(filled["sha256"], update["sha256"])
        self.assertEqual(filled["integrity"], "sha256")
        rebuilt = self.root / "backfill-rebuilt.duckdb"
        rebuild_database(self.staging, rebuilt, reference=self.db)
        self.assertEqual(table_hashes(self.db), table_hashes(rebuilt))

        from src.reproduction.db.verify import verify_files

        (self.corpora / "official" / "train-clean.wav").unlink()
        ok = verify_files(
            self.db,
            staging=self.root / "missing-staging",
            dataset="synthetic.official@1",
            check_hash=True,
            corpora=self.corpora,
            generated=self.generated,
            derived=self.root / "derived",
        )
        self.assertFalse(ok)
        self.assertEqual(
            next(row for row in read_graph(self.db).assets if row["asset_id"] == orphan_id)["sha256"],
            update["sha256"],
        )

    def test_lock_contention_and_incomplete_database(self) -> None:
        lock = self.db.parent / ".reproduction.lock"
        code = (
            "import fcntl, sys, time\n"
            "handle = open(sys.argv[1], 'a+')\n"
            "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(lock)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            line = process.stdout.readline() if process.stdout else ""
            self.assertEqual(line.strip(), "held")
            with self.assertRaises(ReproductionError) as caught:
                initialize_database(self.root / "locked.duckdb")
            self.assertIn("lock", str(caught.exception))
        finally:
            if process.stdout is not None:
                process.stdout.close()
            process.terminate()
            process.wait(timeout=10)

        incomplete = self.root / "incomplete.duckdb"
        import duckdb

        connection = duckdb.connect(str(incomplete))
        create_schema(connection)
        connection.close()
        before = _sha256(incomplete)
        with self.assertRaises(ReproductionError) as caught:
            initialize_database(incomplete)
        self.assertIn("incomplete", str(caught.exception))
        self.assertEqual(before, _sha256(incomplete))

    def test_eval_unknown_without_opening_resource_db(self) -> None:
        safe = fixed_pair_graph(self.corpora, safe=True)
        write_graph(self.staging, safe)
        register_staging(self.staging, self.db)
        leak = [row for row in read_graph(self.db).audits if row["audit_kind"] == "leak_check"]
        self.assertEqual(leak[0]["status"], "UNKNOWN")
        self.assertIn("was not opened", json.dumps(leak[0]["report_json"]))

        empty = self.root / "empty-resource.duckdb"
        import duckdb

        duckdb.connect(str(empty)).close()
        other = self.root / "with-resource.duckdb"
        initialize_database(other)
        other_staging = self.root / "resource-staging"
        write_graph(other_staging, safe)
        register_staging(other_staging, other, resource_db=empty)
        stored = [row for row in read_graph(other).audits if row["audit_kind"] == "leak_check"][0]
        self.assertEqual(stored["status"], "UNKNOWN")
        self.assertIn("eval schema is absent", json.dumps(stored["report_json"]))

        overlap = self.root / "overlap-resource.duckdb"
        connection = duckdb.connect(str(overlap))
        connection.execute("CREATE SCHEMA eval")
        connection.execute("CREATE TABLE eval.cases (recipe_json VARCHAR)")
        sha = safe.assets[0]["sha256"]
        connection.execute(
            "INSERT INTO eval.cases VALUES (?)",
            [json.dumps({"sources": {"clean": {"sha256": sha}}})],
        )
        connection.close()
        rejected = self.root / "rejected.duckdb"
        initialize_database(rejected)
        with self.assertRaises(ReproductionError) as caught:
            register_staging(other_staging, rejected, resource_db=overlap)
        self.assertIn("leak_check", str(caught.exception))
        self.assertEqual([], read_graph(rejected).datasets)

    def test_evaluation_does_not_import_reproduction(self) -> None:
        offenders = []
        for path in (ROOT / "src" / "evaluation").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                if any(name == "src.reproduction" or name.startswith("src.reproduction.") for name in modules):
                    offenders.append(str(path))
        self.assertEqual(offenders, [])
        reproduction = []
        for path in (ROOT / "src" / "reproduction").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                if any(name.startswith(("src.se", "src.asr", "src.evaluation", "src.training")) for name in modules):
                    reproduction.append(f"{path}:{modules}")
        self.assertEqual(reproduction, [])
        script = (
            "import src.evaluation\n"
            "import sys\n"
            "print('src.reproduction' in sys.modules)\n"
            "print('src.reproduction.db' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.strip().splitlines()[-2:], ["False", "False"])

    def test_cli_missing_inputs_and_doctor(self) -> None:
        from src.reproduction.cli import main
        from src.reproduction.doctor import run_doctor

        self.assertEqual(main(["snapshot", "--database", str(self.db), "--dataset", ""]), 1)
        self.assertEqual(main(["rebuild-db", "--database", str(self.db), "--out", ""]), 1)
        self.assertEqual(main(["verify", "--database", str(self.db), "--dataset", "", "--build", ""]), 1)
        resource = self.root / "resource.duckdb"
        import duckdb

        duckdb.connect(str(resource)).close()
        code = run_doctor(
            database=self.root / "missing-reproduction.duckdb",
            resource_db=resource,
            models="",
            corpora=self.corpora,
            generated=self.root / "no-generated",
        )
        self.assertEqual(code, 0)
        incomplete = self.root / "incomplete.duckdb"
        connection = duckdb.connect(str(incomplete))
        create_schema(connection)
        connection.close()
        self.assertEqual(
            run_doctor(database=incomplete, resource_db=resource, models="mp_senet", corpora=self.corpora, generated=self.generated),
            1,
        )

    def test_staging_rejects_different_content(self) -> None:
        graph = fixed_pair_graph(self.corpora, safe=True)
        write_graph(self.staging, graph)
        loaded = load_graph(self.staging)
        self.assertEqual({row["name"] for row in loaded.datasets}, {"synthetic.safe"})
        changed = fixed_pair_graph(self.corpora, safe=True)
        changed.protocols[0]["name"] = "renamed"
        with self.assertRaises(ReproductionError):
            write_graph(self.staging, changed)


def write_and_register(staging: Path, graph: Graph, database: Path) -> None:
    write_graph(staging, graph)
    register_staging(staging, database)


def _dataset_id(database: Path, name: str) -> str:
    rows = [row for row in read_graph(database).datasets if row["name"] == name]
    if len(rows) != 1:
        raise AssertionError(name)
    return rows[0]["dataset_id"]


def _insert_protocol(connection, protocol: dict) -> None:
    connection.execute(
        """
        INSERT INTO repro.protocols VALUES (
            ?, ?, ?, ?, ?, ?, ?::JSON, ?::JSON, ?, ?, ?::JSON
        )
        """,
        [
            protocol["protocol_id"],
            protocol["name"],
            protocol["version"],
            protocol["kind"],
            protocol["algorithm"],
            protocol["algorithm_version"],
            json.dumps(protocol["rule_json"]),
            json.dumps(protocol["official_ref_json"]),
            protocol["deviation_class"],
            protocol["parent_protocol_id"],
            json.dumps(protocol["deviations_json"]),
        ],
    )


if __name__ == "__main__":
    unittest.main()
