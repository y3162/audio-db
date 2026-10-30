import duckdb

from src.data.constants import DUCKDB_MAIN_PATH
from src.data.db.schema import create_corpora_schema


def main() -> None:
    con = duckdb.connect(DUCKDB_MAIN_PATH)
    create_corpora_schema(con, recreate=True)
    con.close()


if __name__ == "__main__":
    main()
