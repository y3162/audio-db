from pathlib import Path

DATA_ROOT = Path("data")
CORPORA_ROOT = DATA_ROOT / "corpora"
DUCKDB_ROOT = DATA_ROOT / "duckdb"
DUCKDB_MAIN_PATH = DUCKDB_ROOT / "resource.duckdb"

INGEST_BATCH_SIZE = 100_000

AUDIO_EXTENSIONS = frozenset({".flac", ".wav"})

SCENE_CLASS_BY_PREFIX = {
    "D": "domestic",
    "N": "nature",
    "O": "office",
    "P": "public",
    "S": "street",
    "T": "transport",
}
