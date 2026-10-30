from pathlib import Path


def load_librispeech_chapter_transcripts(transcript_file: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with open(transcript_file, encoding="utf-8") as handle:
        for line in handle:
            utterance_key, _, text = line.strip().partition(" ")
            if utterance_key:
                transcripts[utterance_key] = text
    return transcripts


def load_libritts_chapter_transcripts(transcript_file: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with open(transcript_file, encoding="utf-8") as handle:
        for line in handle:
            utterance_key, _, rest = line.strip().partition("\t")
            _, _, text = rest.partition("\t")
            if utterance_key:
                transcripts[utterance_key] = text
    return transcripts


def read_transcript_file(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()
