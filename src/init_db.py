import duckdb

from src.constants import DUCKDB_MAIN_PATH
from src.db.schema import create_corpora_schema


def main() -> None:
    con = duckdb.connect(DUCKDB_MAIN_PATH)
    create_corpora_schema(con)
    con.close()


if __name__ == "__main__":
    main()
