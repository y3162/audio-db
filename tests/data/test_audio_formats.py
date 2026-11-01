import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.audio import decode_wav_prefix, read_flac_stream_info, read_wav_stream_info


def _pcm_wav(path: Path, rate: int = 16000, channels: int = 1, frames: int = 32) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x01\x00" * channels * frames)


def _extensible_wav(path: Path) -> None:
    channels = 8
    rate = 16000
    frames = 4
    block = channels * 2
    data = b"\x00\x01" * channels * frames
    fmt = bytearray(40)
    fmt[0:2] = (65534).to_bytes(2, "little")
    fmt[2:4] = channels.to_bytes(2, "little")
    fmt[4:8] = rate.to_bytes(4, "little")
    fmt[8:12] = (rate * block).to_bytes(4, "little")
    fmt[12:14] = block.to_bytes(2, "little")
    fmt[14:16] = (16).to_bytes(2, "little")
    fmt[16:18] = (22).to_bytes(2, "little")
    fmt[18:20] = (16).to_bytes(2, "little")
    fmt[24:40] = bytes.fromhex("0100000000001000800000aa00389b71")
    body = b"fmt " + len(fmt).to_bytes(4, "little") + bytes(fmt) + b"data" + len(data).to_bytes(4, "little") + data
    payload = b"WAVE" + body
    path.write_bytes(b"RIFF" + len(payload).to_bytes(4, "little") + payload)


class AudioFormatTests(unittest.TestCase):
    def test_pcm_wav_and_reject_garbage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "ok.wav"
            _pcm_wav(wav, frames=20)
            self.assertEqual(read_wav_stream_info(wav), (16000, 20, 1))
            self.assertEqual(decode_wav_prefix(wav, 4)[2], 4)
            bad = root / "bad.wav"
            bad.write_bytes(b"not-a-wav")
            with self.assertRaises(ValueError):
                read_wav_stream_info(bad)

    def test_extensible_pcm(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ext.wav"
            _extensible_wav(path)
            self.assertEqual(read_wav_stream_info(path), (16000, 4, 8))
            rate, channels, frames = decode_wav_prefix(path, 4)
            self.assertEqual((rate, channels, frames), (16000, 8, 4))

    def test_flac_header(self) -> None:
        sample = Path("data/corpora/LibriSpeech")
        if not sample.is_dir():
            self.skipTest("LibriSpeech is not linked")
        flac = next(sample.rglob("*.flac"))
        rate, frames, channels = read_flac_stream_info(flac)
        self.assertGreater(rate, 0)
        self.assertGreater(frames, 0)
        self.assertGreater(channels, 0)

    def test_real_openslr28_extensible(self) -> None:
        folder = Path("data/corpora/OpenSLR28/RIRS_NOISES/real_rirs_isotropic_noises")
        if not folder.is_dir():
            self.skipTest("OpenSLR28 is not linked")
        wavs = sorted(folder.glob("*.wav"))
        self.assertGreaterEqual(len(wavs), 272)
        failures = []
        for path in wavs:
            try:
                rate, frames, channels = read_wav_stream_info(path)
                decode_wav_prefix(path, 2)
            except ValueError as exc:
                failures.append((path.name, str(exc)))
                continue
            if rate < 1 or frames < 1 or channels < 1:
                failures.append((path.name, "empty stream"))
        self.assertEqual(failures, [])
