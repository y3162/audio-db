import os
from pathlib import Path
from typing import BinaryIO

# WAVEFORMATEXTENSIBLE subformats (GUID, little-endian layout).
_PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
_FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")


def read_audio_stream_info(audio_path: Path) -> tuple[int, int, int]:
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return read_wav_stream_info(audio_path)
    if suffix == ".flac":
        return read_flac_stream_info(audio_path)
    raise ValueError(f"Unsupported audio format: {audio_path}")


def read_wav_stream_info(audio_path: Path) -> tuple[int, int, int]:
    with open(audio_path, "rb", buffering=0) as f:
        _advise_random(f)
        header = f.read(44)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Not a WAV file: {audio_path}")
        if (
            len(header) >= 44
            and header[12:16] == b"fmt "
            and int.from_bytes(header[16:20], "little") == 16
            and header[36:40] == b"data"
        ):
            audio_format = int.from_bytes(header[20:22], "little")
            channels = int.from_bytes(header[22:24], "little")
            sample_rate = int.from_bytes(header[24:28], "little")
            block_align = int.from_bytes(header[32:34], "little")
            data_size = int.from_bytes(header[40:44], "little")
            fmt_data = None
        else:
            f.seek(12)
            audio_format, channels, sample_rate, block_align, data_size, fmt_data = _read_wav_chunks(f, audio_path)
        _accept_wav_format(audio_format, fmt_data, audio_path)

    if block_align == 0:
        raise ValueError(f"Invalid WAV block align: {audio_path}")
    if channels < 1 or sample_rate < 1:
        raise ValueError(f"Invalid WAV stream info: {audio_path}")
    return sample_rate, data_size // block_align, channels


def read_flac_stream_info(audio_path: Path) -> tuple[int, int, int]:
    with open(audio_path, "rb", buffering=0) as f:
        _advise_random(f)
        header = f.read(42)
    if len(header) < 42 or header[:4] != b"fLaC":
        raise ValueError(f"Not a FLAC file: {audio_path}")

    # | sample rate | channels-1 | bits/sample-1 | total samples |
    # |   20 bits   |   3 bits   |    5 bits     |    36 bits    |
    packed = int.from_bytes(header[18:26], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    frames = packed & 0xFFFFFFFFF
    return sample_rate, frames, channels


def _advise_random(f: BinaryIO) -> None:
    try:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
    except OSError:
        pass


def _accept_wav_format(audio_format: int, fmt_data: bytes | None, audio_path: Path) -> None:
    """Accept PCM, IEEE float, and extensible WAV whose subformat is one of those."""
    if audio_format in (1, 3):
        return
    if audio_format != 65534:
        raise ValueError(f"Unsupported WAV format {audio_format}: {audio_path}")
    if fmt_data is None or len(fmt_data) < 40:
        raise ValueError(f"WAVE_FORMAT_EXTENSIBLE fmt chunk is incomplete: {audio_path}")
    guid = fmt_data[24:40]
    if guid not in (_PCM_GUID, _FLOAT_GUID):
        raise ValueError(f"Unsupported WAV extensible subformat {guid.hex()}: {audio_path}")


def decode_wav_prefix(audio_path: Path, frames: int = 16) -> tuple[int, int, int]:
    """Read a short PCM prefix. Returns sample rate, channels, and decoded frame count."""
    sample_rate, total, channels = read_wav_stream_info(audio_path)
    with open(audio_path, "rb", buffering=0) as handle:
        handle.seek(12)
        bits = None
        data = None
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id = header[:4]
            chunk_size = int.from_bytes(header[4:8], "little")
            if chunk_id == b"fmt ":
                fmt = handle.read(chunk_size)
                bits = int.from_bytes(fmt[14:16], "little")
                if chunk_size % 2:
                    handle.seek(1, 1)
            elif chunk_id == b"data":
                data = handle.read(min(chunk_size, frames * channels * max(bits or 16, 16) // 8))
                break
            else:
                handle.seek(chunk_size + (chunk_size % 2), 1)
    if bits not in (16, 32) or data is None:
        raise ValueError(f"Cannot decode WAV samples ({bits} bit): {audio_path}")
    width = bits // 8
    count = len(data) // (width * channels)
    if count < 1:
        raise ValueError(f"WAV data chunk is empty: {audio_path}")
    if bits == 16:
        for index in range(count * channels):
            int.from_bytes(data[index * 2 : index * 2 + 2], "little", signed=True)
    return sample_rate, channels, min(count, total)


def _read_wav_chunks(f: BinaryIO, audio_path: Path) -> tuple[int, int, int, int, int, bytes]:
    fmt = None
    fmt_data_kept = b""
    data_size = None
    while True:
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id = chunk_header[:4]
        chunk_size = int.from_bytes(chunk_header[4:8], "little")
        if chunk_id == b"fmt ":
            fmt_data = f.read(chunk_size)
            if len(fmt_data) < 16:
                raise ValueError(f"Invalid WAV fmt chunk: {audio_path}")
            audio_format = int.from_bytes(fmt_data[0:2], "little")
            channels = int.from_bytes(fmt_data[2:4], "little")
            sample_rate = int.from_bytes(fmt_data[4:8], "little")
            block_align = int.from_bytes(fmt_data[12:14], "little")
            fmt = (audio_format, channels, sample_rate, block_align)
            fmt_data_kept = fmt_data
            if chunk_size % 2:
                f.seek(1, 1)
        elif chunk_id == b"data":
            data_size = chunk_size
        else:
            f.seek(chunk_size, 1)
            if chunk_size % 2:
                f.seek(1, 1)
        if fmt is not None and data_size is not None:
            break

    if fmt is None or data_size is None:
        raise ValueError(f"Missing WAV fmt or data chunk: {audio_path}")
    audio_format, channels, sample_rate, block_align = fmt
    return audio_format, channels, sample_rate, block_align, data_size, fmt_data_kept
