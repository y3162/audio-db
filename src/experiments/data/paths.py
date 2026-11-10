"""Default locations for the reproduction database.

These paths are the production locations. Tests pass their own directories.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DEFAULT_DB = DATA / "db" / "reproduction.duckdb"
DEFAULT_STAGING = DATA / "staging" / "reproduction"
DEFAULT_SNAPSHOTS = DATA / "db" / "snapshots"
DEFAULT_CORPORA = DATA / "corpora"
DEFAULT_GENERATED = DATA / "generated"
DEFAULT_DERIVED = DATA / "derived"
DEFAULT_RESOURCE_DB = DATA / "db" / "resource.duckdb"
LOCK_NAME = ".reproduction.lock"
