"""Evaluation tables. These statements never touch corpora.audio_files.

The schema is named eval. Do not put the database file itself at a path whose
stem is eval: DuckDB uses that stem as the catalog name and then rejects
eval.cases as ambiguous.

Cases store a recipe document. Generation parameters are not separate columns.
"""

from __future__ import annotations


def create_eval_schema(con) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS eval;")
    existing = con.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'eval' AND table_name = 'cases'
        """
    ).fetchall()
    if existing:
        names = {row[0] for row in existing}
        if "recipe_json" not in names:
            raise ValueError(
                "eval.cases is the previous fixed-column schema. Refusing to alter this database."
            )
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.cases (
            case_id VARCHAR PRIMARY KEY,
            kind VARCHAR NOT NULL CHECK (kind IN ('clean', 'noisy', 'paired', 'synthetic')),
            dataset VARCHAR NOT NULL,
            split VARCHAR,
            sample_rate INTEGER,
            transcript VARCHAR,
            speaker_id VARCHAR,
            recipe_json JSON NOT NULL,
            content_hash VARCHAR NOT NULL,
            metadata JSON NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.sets (
            name VARCHAR PRIMARY KEY,
            description VARCHAR,
            content_hash VARCHAR NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.set_items (
            set_name VARCHAR NOT NULL,
            position INTEGER NOT NULL,
            case_id VARCHAR NOT NULL,
            PRIMARY KEY (set_name, position),
            UNIQUE (set_name, case_id),
            FOREIGN KEY (set_name) REFERENCES eval.sets(name),
            FOREIGN KEY (case_id) REFERENCES eval.cases(case_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.runs (
            run_id VARCHAR PRIMARY KEY,
            config_hash VARCHAR NOT NULL,
            config_json JSON NOT NULL,
            content_hash VARCHAR NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.results (
            record_id VARCHAR PRIMARY KEY,
            run_id VARCHAR NOT NULL,
            case_id VARCHAR,
            payload_json JSON NOT NULL,
            content_hash VARCHAR NOT NULL,
            FOREIGN KEY (run_id) REFERENCES eval.runs(run_id)
        )
    """)
