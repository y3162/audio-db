"""Evaluation tables. These statements never touch corpora.audio_files.

The schema is named eval. Do not put the database file itself at a path whose
stem is eval: DuckDB uses that stem as the catalog name and then rejects
eval.cases as ambiguous.
"""


def create_eval_schema(con) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS eval;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.cases (
            case_id VARCHAR PRIMARY KEY,
            kind VARCHAR NOT NULL CHECK (kind IN ('clean', 'noisy', 'paired', 'synthetic')),
            dataset VARCHAR NOT NULL,
            split VARCHAR,
            sample_rate INTEGER,
            transcript VARCHAR,
            speaker_id VARCHAR,
            snr_db DOUBLE,
            audio_path VARCHAR,
            clean_path VARCHAR,
            noisy_path VARCHAR,
            noise_path VARCHAR,
            corpus_path VARCHAR,
            mix_protocol VARCHAR,
            mix_seed BIGINT,
            noise_start BIGINT,
            max_seconds DOUBLE,
            prevent_clip BOOLEAN,
            hard_clip BOOLEAN,
            metadata VARCHAR NOT NULL,
            content_hash VARCHAR NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval.sets (
            name VARCHAR PRIMARY KEY,
            description VARCHAR
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
