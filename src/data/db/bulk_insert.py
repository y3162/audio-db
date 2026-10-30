import csv
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence

import duckdb

from src.data.constants import INGEST_BATCH_SIZE
from src.data.models.audio_file import AUDIO_FILE_COLUMNS, AudioFileRow


def copy_insert_rows(con: duckdb.DuckDBPyConnection, rows: Sequence[AudioFileRow]) -> None:
    if not rows:
        return
    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, lineterminator="\n")
            writer.writerow(AUDIO_FILE_COLUMNS)
            for row in rows:
                writer.writerow("\\N" if value is None else value for value in row.values())
        column_list = ", ".join(AUDIO_FILE_COLUMNS)
        escaped_path = csv_path.replace("'", "''")
        con.execute(
            f"COPY corpora.audio_files ({column_list}) FROM '{escaped_path}' "
            "(HEADER TRUE, DELIMITER ',', QUOTE '\"', ESCAPE '\"', NULL '\\N')"
        )
    finally:
        os.unlink(csv_path)


def copy_insert_batched(
    con: duckdb.DuckDBPyConnection,
    rows: Iterable[AudioFileRow],
    *,
    batch_size: int = INGEST_BATCH_SIZE,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    total = 0
    batch: list[AudioFileRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            copy_insert_rows(con, batch)
            total += len(batch)
            if on_progress is not None:
                on_progress(total)
            batch.clear()
    if batch:
        copy_insert_rows(con, batch)
        total += len(batch)
        if on_progress is not None:
            on_progress(total)
    return total
