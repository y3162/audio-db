"""``repro`` schema for reproduction.duckdb.

Column order here is the order stored in parquet and inserted into DuckDB.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import SCHEMA_VERSION, digest

TABLE_ORDER = (
    "schema_info",
    "catalog_snapshots",
    "derivations",
    "builds",
    "assets",
    "material_sets",
    "material_set_members",
    "protocols",
    "cases",
    "case_sets",
    "case_set_items",
    "datasets",
    "dataset_bindings",
    "audits",
    "runs",
)


@dataclass(frozen=True)
class TableSpec:
    name: str
    pk: tuple[str, ...]
    columns: tuple[str, ...]
    json_columns: tuple[str, ...] = ()


TABLES: dict[str, TableSpec] = {
    "schema_info": TableSpec("schema_info", ("key",), ("key", "value")),
    "catalog_snapshots": TableSpec(
        "catalog_snapshots",
        ("catalog_id",),
        ("catalog_id", "source_kind", "corpora_json", "row_count", "source_db_sha256"),
        ("corpora_json",),
    ),
    "derivations": TableSpec(
        "derivations",
        ("derivation_id",),
        ("derivation_id", "kind", "spec_json"),
        ("spec_json",),
    ),
    "builds": TableSpec(
        "builds",
        ("build_id",),
        (
            "build_id",
            "generator",
            "generator_version",
            "params_json",
            "seed",
            "inputs_json",
            "shard_size",
        ),
        ("params_json", "inputs_json"),
    ),
    "assets": TableSpec(
        "assets",
        ("asset_id",),
        (
            "asset_id",
            "storage_root",
            "rel_path",
            "corpus",
            "subset",
            "speaker_id",
            "utterance_key",
            "sample_rate",
            "frames",
            "channels",
            "sample_format",
            "size_bytes",
            "sha256",
            "integrity",
            "catalog_id",
            "derivation_id",
            "derived_from",
            "build_id",
            "extra_json",
        ),
        ("extra_json",),
    ),
    "material_sets": TableSpec(
        "material_sets",
        ("material_set_id",),
        (
            "material_set_id",
            "name",
            "version",
            "role",
            "split_role",
            "origin",
            "selection_rule_json",
            "provenance_json",
            "member_count",
            "total_seconds",
            "list_sha256",
            "integrity_min",
        ),
        ("selection_rule_json", "provenance_json"),
    ),
    "material_set_members": TableSpec(
        "material_set_members",
        ("material_set_id", "position"),
        ("material_set_id", "position", "asset_id", "group_key", "attrs_json"),
        ("attrs_json",),
    ),
    "protocols": TableSpec(
        "protocols",
        ("protocol_id",),
        (
            "protocol_id",
            "name",
            "version",
            "kind",
            "algorithm",
            "algorithm_version",
            "rule_json",
            "official_ref_json",
            "deviation_class",
            "parent_protocol_id",
            "deviations_json",
        ),
        ("rule_json", "official_ref_json", "deviations_json"),
    ),
    "cases": TableSpec(
        "cases",
        ("case_id",),
        (
            "case_id",
            "doc_kind",
            "doc_json",
            "case_role",
            "protocol_id",
            "clean_asset_id",
            "noisy_asset_id",
            "noise_asset_id",
            "teacher_kind",
            "sample_rate",
            "frames",
            "speaker_id",
            "utterance_key",
            "transcript",
            "build_id",
            "gen_index",
            "content_hash",
        ),
        ("doc_json",),
    ),
    "case_sets": TableSpec(
        "case_sets",
        ("case_set_id",),
        (
            "case_set_id",
            "name",
            "version",
            "split_role",
            "origin",
            "selection_rule_json",
            "provenance_json",
            "member_count",
            "list_sha256",
        ),
        ("selection_rule_json", "provenance_json"),
    ),
    "case_set_items": TableSpec(
        "case_set_items",
        ("case_set_id", "position"),
        ("case_set_id", "position", "case_id"),
    ),
    "datasets": TableSpec(
        "datasets",
        ("dataset_id",),
        (
            "dataset_id",
            "name",
            "version",
            "protocol_id",
            "profile",
            "material_origin",
            "claim_label",
            "declared_status",
            "blocking_json",
            "selection_uses_test",
            "selection_note",
            "deviations_json",
        ),
        ("blocking_json", "deviations_json"),
    ),
    "dataset_bindings": TableSpec(
        "dataset_bindings",
        ("dataset_id", "slot"),
        ("dataset_id", "slot", "case_set_id", "material_set_id", "params_json"),
        ("params_json",),
    ),
    "audits": TableSpec(
        "audits",
        ("audit_id",),
        ("audit_id", "subject_kind", "subject_id", "audit_kind", "status", "report_json"),
        ("report_json",),
    ),
    "runs": TableSpec(
        "runs",
        ("run_id",),
        (
            "run_id",
            "dataset_id",
            "snapshot_hash",
            "config_hash",
            "git_commit",
            "world_size",
            "ddp_mode",
            "selection_uses_test",
            "data_state_json",
        ),
        ("data_state_json",),
    ),
}

INSERT_ORDER = (
    "catalog_snapshots",
    "derivations",
    "builds",
    "assets",
    "material_sets",
    "material_set_members",
    "protocols",
    "cases",
    "case_sets",
    "case_set_items",
    "datasets",
    "dataset_bindings",
    "audits",
    "runs",
)

_STATEMENTS = (
    "CREATE SCHEMA IF NOT EXISTS repro;",
    """
    CREATE TABLE repro.schema_info (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE repro.catalog_snapshots (
        catalog_id VARCHAR PRIMARY KEY,
        source_kind VARCHAR NOT NULL CHECK (source_kind IN ('resource_duckdb', 'staging_parquet')),
        corpora_json JSON NOT NULL,
        row_count BIGINT NOT NULL CHECK (row_count >= 0),
        source_db_sha256 VARCHAR
    )
    """,
    """
    CREATE TABLE repro.derivations (
        derivation_id VARCHAR PRIMARY KEY,
        kind VARCHAR NOT NULL CHECK (kind IN ('resample', 'format_convert')),
        spec_json JSON NOT NULL
    )
    """,
    """
    CREATE TABLE repro.builds (
        build_id VARCHAR PRIMARY KEY,
        generator VARCHAR NOT NULL,
        generator_version INTEGER NOT NULL CHECK (generator_version > 0),
        params_json JSON NOT NULL,
        seed BIGINT NOT NULL,
        inputs_json JSON NOT NULL,
        shard_size INTEGER NOT NULL CHECK (shard_size > 0)
    )
    """,
    """
    CREATE TABLE repro.assets (
        asset_id VARCHAR PRIMARY KEY,
        storage_root VARCHAR NOT NULL CHECK (storage_root IN ('corpora', 'generated', 'derived')),
        rel_path VARCHAR NOT NULL,
        corpus VARCHAR,
        subset VARCHAR,
        speaker_id VARCHAR,
        utterance_key VARCHAR,
        sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
        frames BIGINT NOT NULL CHECK (frames > 0),
        channels INTEGER NOT NULL CHECK (channels > 0),
        sample_format VARCHAR,
        size_bytes BIGINT CHECK (size_bytes IS NULL OR size_bytes >= 0),
        sha256 VARCHAR,
        integrity VARCHAR NOT NULL CHECK (integrity IN ('sha256', 'size', 'catalog')),
        catalog_id VARCHAR,
        derivation_id VARCHAR,
        derived_from VARCHAR,
        build_id VARCHAR,
        extra_json JSON NOT NULL,
        CHECK (asset_id = storage_root || '/' || rel_path),
        CHECK (
            (integrity = 'sha256' AND sha256 IS NOT NULL)
            OR (integrity = 'size' AND sha256 IS NULL AND size_bytes IS NOT NULL)
            OR (integrity = 'catalog' AND sha256 IS NULL)
        ),
        CHECK (storage_root <> 'corpora' OR catalog_id IS NOT NULL),
        CHECK (storage_root <> 'derived' OR (derivation_id IS NOT NULL AND derived_from IS NOT NULL)),
        CHECK (storage_root <> 'generated' OR build_id IS NOT NULL),
        FOREIGN KEY (catalog_id) REFERENCES repro.catalog_snapshots(catalog_id),
        FOREIGN KEY (derivation_id) REFERENCES repro.derivations(derivation_id),
        FOREIGN KEY (derived_from) REFERENCES repro.assets(asset_id),
        FOREIGN KEY (build_id) REFERENCES repro.builds(build_id)
    )
    """,
    """
    CREATE TABLE repro.material_sets (
        material_set_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        version INTEGER NOT NULL CHECK (version > 0),
        role VARCHAR NOT NULL CHECK (role IN ('speech', 'noise', 'rir')),
        split_role VARCHAR NOT NULL CHECK (split_role IN ('train', 'valid', 'test', 'pool')),
        origin VARCHAR NOT NULL CHECK (origin IN ('official', 'substitute', 'derived_from_official', 'unknown')),
        selection_rule_json JSON NOT NULL,
        provenance_json JSON NOT NULL,
        member_count BIGINT NOT NULL CHECK (member_count >= 0),
        total_seconds DOUBLE NOT NULL CHECK (total_seconds >= 0),
        list_sha256 VARCHAR NOT NULL,
        integrity_min VARCHAR NOT NULL CHECK (integrity_min IN ('sha256', 'size', 'catalog')),
        UNIQUE (name, version)
    )
    """,
    """
    CREATE TABLE repro.material_set_members (
        material_set_id VARCHAR NOT NULL,
        position INTEGER NOT NULL CHECK (position >= 0),
        asset_id VARCHAR NOT NULL,
        group_key VARCHAR,
        attrs_json JSON,
        PRIMARY KEY (material_set_id, position),
        UNIQUE (material_set_id, asset_id),
        FOREIGN KEY (material_set_id) REFERENCES repro.material_sets(material_set_id),
        FOREIGN KEY (asset_id) REFERENCES repro.assets(asset_id)
    )
    """,
    """
    CREATE TABLE repro.protocols (
        protocol_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        version INTEGER NOT NULL CHECK (version > 0),
        kind VARCHAR NOT NULL CHECK (kind IN ('fixed_pair', 'pregenerated', 'online_mixing')),
        algorithm VARCHAR NOT NULL,
        algorithm_version INTEGER NOT NULL CHECK (algorithm_version > 0),
        rule_json JSON NOT NULL,
        official_ref_json JSON NOT NULL,
        deviation_class VARCHAR NOT NULL CHECK (
            deviation_class IN (
                'official_effective', 'minimal_launch_repair', 'algorithm_change', 'research_improvement'
            )
        ),
        parent_protocol_id VARCHAR,
        deviations_json JSON NOT NULL,
        UNIQUE (name, version),
        CHECK (deviation_class = 'official_effective' OR parent_protocol_id IS NOT NULL),
        FOREIGN KEY (parent_protocol_id) REFERENCES repro.protocols(protocol_id)
    )
    """,
    """
    CREATE TABLE repro.cases (
        case_id VARCHAR PRIMARY KEY,
        doc_kind VARCHAR NOT NULL CHECK (doc_kind IN ('recipe_v1', 'realization_v1')),
        doc_json JSON NOT NULL,
        case_role VARCHAR NOT NULL CHECK (case_role IN ('recorded_pair', 'generated', 'online_snapshot')),
        protocol_id VARCHAR NOT NULL,
        clean_asset_id VARCHAR,
        noisy_asset_id VARCHAR,
        noise_asset_id VARCHAR,
        teacher_kind VARCHAR NOT NULL CHECK (
            teacher_kind IN ('dry_clean', 'reverberant_clean', 'official_mixed_target', 'none')
        ),
        sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
        frames BIGINT CHECK (frames IS NULL OR frames > 0),
        speaker_id VARCHAR,
        utterance_key VARCHAR,
        transcript VARCHAR,
        build_id VARCHAR,
        gen_index BIGINT CHECK (gen_index IS NULL OR gen_index >= 0),
        content_hash VARCHAR NOT NULL,
        CHECK (
            case_role <> 'recorded_pair'
            OR (
                clean_asset_id IS NOT NULL AND noisy_asset_id IS NOT NULL
                AND teacher_kind = 'dry_clean' AND doc_kind = 'recipe_v1'
            )
        ),
        CHECK (case_role <> 'generated' OR (build_id IS NOT NULL AND gen_index IS NOT NULL)),
        CHECK (
            case_role <> 'online_snapshot'
            OR (
                clean_asset_id IS NULL AND noisy_asset_id IS NULL AND noise_asset_id IS NULL
                AND build_id IS NULL AND gen_index IS NULL
            )
        ),
        FOREIGN KEY (protocol_id) REFERENCES repro.protocols(protocol_id),
        FOREIGN KEY (clean_asset_id) REFERENCES repro.assets(asset_id),
        FOREIGN KEY (noisy_asset_id) REFERENCES repro.assets(asset_id),
        FOREIGN KEY (noise_asset_id) REFERENCES repro.assets(asset_id),
        FOREIGN KEY (build_id) REFERENCES repro.builds(build_id)
    )
    """,
    """
    CREATE TABLE repro.case_sets (
        case_set_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        version INTEGER NOT NULL CHECK (version > 0),
        split_role VARCHAR NOT NULL CHECK (split_role IN ('train', 'valid', 'test')),
        origin VARCHAR NOT NULL CHECK (origin IN ('official', 'substitute', 'derived_from_official', 'unknown')),
        selection_rule_json JSON NOT NULL,
        provenance_json JSON NOT NULL,
        member_count BIGINT NOT NULL CHECK (member_count >= 0),
        list_sha256 VARCHAR NOT NULL,
        UNIQUE (name, version)
    )
    """,
    """
    CREATE TABLE repro.case_set_items (
        case_set_id VARCHAR NOT NULL,
        position INTEGER NOT NULL CHECK (position >= 0),
        case_id VARCHAR NOT NULL,
        PRIMARY KEY (case_set_id, position),
        UNIQUE (case_set_id, case_id),
        FOREIGN KEY (case_set_id) REFERENCES repro.case_sets(case_set_id),
        FOREIGN KEY (case_id) REFERENCES repro.cases(case_id)
    )
    """,
    """
    CREATE TABLE repro.datasets (
        dataset_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        version INTEGER NOT NULL CHECK (version > 0),
        protocol_id VARCHAR NOT NULL,
        profile VARCHAR NOT NULL CHECK (profile IN ('official_faithful', 'leakage_safe')),
        material_origin VARCHAR NOT NULL CHECK (
            material_origin IN ('official', 'derived_from_official', 'substitute', 'mixed', 'unknown')
        ),
        claim_label VARCHAR NOT NULL CHECK (
            claim_label IN (
                'official_reproduction',
                'protocol_conformant_substitute_materials',
                'espnet_public_se_recipe_reproduction',
                'official_recipe_scaled_down',
                'minimal_repair_reproduction',
                'leakage_safe_comparison',
                'research_checkpoint_selection'
            )
        ),
        declared_status VARCHAR NOT NULL CHECK (declared_status IN ('ready', 'blocked', 'unknown')),
        blocking_json JSON NOT NULL,
        selection_uses_test BOOLEAN NOT NULL,
        selection_note VARCHAR,
        deviations_json JSON NOT NULL,
        UNIQUE (name, version),
        CHECK (profile <> 'leakage_safe' OR selection_uses_test = false),
        CHECK (profile <> 'leakage_safe' OR claim_label = 'leakage_safe_comparison'),
        CHECK (claim_label <> 'leakage_safe_comparison' OR profile = 'leakage_safe'),
        CHECK (claim_label <> 'official_reproduction' OR profile = 'official_faithful'),
        FOREIGN KEY (protocol_id) REFERENCES repro.protocols(protocol_id)
    )
    """,
    """
    CREATE TABLE repro.dataset_bindings (
        dataset_id VARCHAR NOT NULL,
        slot VARCHAR NOT NULL CHECK (
            slot IN (
                'train.cases', 'valid.cases', 'test.cases',
                'train.speech', 'train.noise', 'train.rir',
                'valid.speech', 'valid.noise', 'valid.rir',
                'test.speech', 'test.noise', 'test.rir'
            )
        ),
        case_set_id VARCHAR,
        material_set_id VARCHAR,
        params_json JSON,
        PRIMARY KEY (dataset_id, slot),
        CHECK ((case_set_id IS NULL) <> (material_set_id IS NULL)),
        FOREIGN KEY (dataset_id) REFERENCES repro.datasets(dataset_id),
        FOREIGN KEY (case_set_id) REFERENCES repro.case_sets(case_set_id),
        FOREIGN KEY (material_set_id) REFERENCES repro.material_sets(material_set_id)
    )
    """,
    """
    CREATE TABLE repro.audits (
        audit_id VARCHAR PRIMARY KEY,
        subject_kind VARCHAR NOT NULL CHECK (subject_kind IN ('build', 'dataset', 'material_set', 'case_set')),
        subject_id VARCHAR NOT NULL,
        audit_kind VARCHAR NOT NULL CHECK (
            audit_kind IN (
                'build_verification', 'count_check', 'list_hash_check',
                'leak_check', 'distribution_check', 'hash_verify'
            )
        ),
        status VARCHAR NOT NULL CHECK (status IN ('PASS', 'FAIL', 'WARN', 'UNKNOWN')),
        report_json JSON NOT NULL
    )
    """,
    """
    CREATE TABLE repro.runs (
        run_id VARCHAR PRIMARY KEY,
        dataset_id VARCHAR NOT NULL,
        snapshot_hash VARCHAR NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR NOT NULL,
        world_size INTEGER NOT NULL CHECK (world_size > 0),
        ddp_mode VARCHAR NOT NULL CHECK (ddp_mode IN ('single', 'real', 'virtual')),
        selection_uses_test BOOLEAN NOT NULL,
        data_state_json JSON NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES repro.datasets(dataset_id)
    )
    """,
    "CREATE INDEX repro_assets_corpus ON repro.assets(corpus, subset)",
    "CREATE INDEX repro_assets_sha256 ON repro.assets(sha256)",
    "CREATE INDEX repro_assets_speaker ON repro.assets(speaker_id)",
    "CREATE INDEX repro_assets_build ON repro.assets(build_id)",
    "CREATE INDEX repro_cases_build ON repro.cases(build_id, gen_index)",
    "CREATE INDEX repro_cases_clean ON repro.cases(clean_asset_id)",
    "CREATE INDEX repro_cases_speaker ON repro.cases(speaker_id)",
)


def expected_columns() -> list[list[str]]:
    """Column order matches information_schema: table name, then ordinal."""
    pairs = []
    for name in sorted(TABLES):
        for column in TABLES[name].columns:
            pairs.append([name, column])
    return pairs


def column_fingerprint() -> str:
    return digest(expected_columns())


def _create_statement(table: str) -> str:
    needle = f"CREATE TABLE repro.{table} ("
    for statement in _STATEMENTS:
        if needle in statement:
            return statement
    raise ReproductionError(f"missing CREATE TABLE for repro.{table}")


def _table_exists(connection, table: str) -> bool:
    found = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'repro' AND table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(found and found[0])


def _column_names(connection, table: str) -> list[str]:
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'repro' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table],
    ).fetchall()
    return [row[0] for row in rows]


def ensure_dataset_claim_labels(connection) -> None:
    """Allow ``research_checkpoint_selection`` on an existing datasets table.

    DuckDB cannot drop a CHECK in place. The datasets table and the bindings
    that reference it are copied, recreated from the current CREATE TABLE, and
    copied back. Row values are not edited. Column order stays the same.
    """
    rows = connection.execute(
        """
        SELECT constraint_text
        FROM duckdb_constraints()
        WHERE schema_name = 'repro'
          AND table_name = 'datasets'
          AND constraint_text IS NOT NULL
        """
    ).fetchall()
    if any("research_checkpoint_selection" in str(row[0]) for row in rows):
        return
    if not _table_exists(connection, "datasets"):
        raise ReproductionError("repro.datasets was not found; the database was not modified")
    dependents = [
        row[0]
        for row in connection.execute(
            """
            SELECT DISTINCT table_name
            FROM duckdb_constraints()
            WHERE schema_name = 'repro' AND referenced_table = 'datasets'
            """
        ).fetchall()
        if row[0] != "datasets"
    ]
    dataset_columns = _column_names(connection, "datasets")
    saved = {name: _column_names(connection, name) for name in dependents}
    connection.execute("CREATE TABLE repro.datasets__claim_keep AS SELECT * FROM repro.datasets")
    for name in dependents:
        connection.execute(f"CREATE TABLE repro.{name}__claim_keep AS SELECT * FROM repro.{name}")
        connection.execute(f"DROP TABLE repro.{name}")
    connection.execute("DROP TABLE repro.datasets")
    connection.execute(_create_statement("datasets"))
    dataset_sql = ", ".join(dataset_columns)
    connection.execute(
        f"INSERT INTO repro.datasets ({dataset_sql}) SELECT {dataset_sql} FROM repro.datasets__claim_keep"
    )
    connection.execute("DROP TABLE repro.datasets__claim_keep")
    for name in dependents:
        connection.execute(_create_statement(name))
        column_sql = ", ".join(saved[name])
        connection.execute(
            f"INSERT INTO repro.{name} ({column_sql}) SELECT {column_sql} FROM repro.{name}__claim_keep"
        )
        connection.execute(f"DROP TABLE repro.{name}__claim_keep")


def create_schema(connection) -> None:
    for statement in _STATEMENTS:
        connection.execute(statement)


def mark_ready(connection) -> None:
    rows = (
        ("schema_version", SCHEMA_VERSION),
        ("ready", "1"),
        ("column_fingerprint", column_fingerprint()),
    )
    connection.executemany("INSERT INTO repro.schema_info (key, value) VALUES (?, ?)", list(rows))


def actual_columns(connection) -> list[list[str]]:
    rows = connection.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'repro'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    return [[row[0], row[1]] for row in rows]


def assert_ready(connection) -> None:
    try:
        found = actual_columns(connection)
    except Exception as exc:
        raise ReproductionError(f"database is not a reproduction database: {exc}") from exc
    expected = expected_columns()
    if found != expected:
        raise ReproductionError(
            "reproduction schema does not match this code; the database was not modified"
        )
    info = {
        row[0]: row[1]
        for row in connection.execute("SELECT key, value FROM repro.schema_info").fetchall()
    }
    if info.get("schema_version") != SCHEMA_VERSION or info.get("ready") != "1":
        raise ReproductionError("reproduction database is incomplete (ready flag is missing)")
    if info.get("column_fingerprint") != column_fingerprint():
        raise ReproductionError("reproduction database fingerprint does not match this code")
