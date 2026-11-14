"""SEMamba++ generator, MRD, and CQT discriminator for the experiment framework.

``SEMambaTask`` builds the generator, then the multi-resolution discriminator,
then the CQT discriminator inside one ``seed_model_init`` scope. Builder stage
0 does not keep that order. Each class therefore enters ``seed_model_init``
alone and replays the draws it does not keep. Discarded modules are not
registered and are not optimizer targets.

The generator is ``src.se.build_model("semambapp")``. The discriminators are
the existing classes in ``src.experiments.official.semambapp.discriminator``. Neither network is
reimplemented here. ``scan_backend`` is a runtime switch on that generator,
not a second architecture.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.se import build_model
from src.se.models.semambapp.network import SEMambaPP
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.errors import TrainingError
from src.experiments.official.semambapp.discriminator import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator
from src.experiments.calc.stft import semamba_mag_phase_istft, semamba_mag_phase_stft

_SAMPLE_RATE = 16000
_COMPRESS = 0.3
_MRD_RESOLUTIONS = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))
_CQT_HOPS = (512, 256, 256)
_CQT_OCTAVES = (8, 8, 8)
_CQT_BINS = (12, 24, 36)
_DISCRIMINATOR_KEYS = frozenset(
    {
        "cqt_bins_per_octave",
        "cqt_channels",
        "cqt_hops",
        "cqt_octaves",
        "mrd_channels",
        "mrd_resolutions",
    }
)
_FORWARD_FIELDS = (
    "clean_mag",
    "clean_phase",
    "clean_complex",
    "mag",
    "phase",
    "complex",
    "recomputed",
    "audio",
)


def architecture_config(
    *,
    hid_feature: int,
    num_tfmamba: int,
    d_state: int,
    d_conv: int,
    expand: int,
    norm_epsilon: float,
    input_channel: int,
    output_channel: int,
    n_fft: int,
    scan_backend: str,
    hop: int,
    win: int,
    sampling_rate: int,
    mrd_channels: int = 32,
    mrd_resolutions: tuple[tuple[int, int, int], ...] | list[list[int]] = _MRD_RESOLUTIONS,
    cqt_channels: int = 128,
    cqt_hops: tuple[int, ...] | list[int] = _CQT_HOPS,
    cqt_octaves: tuple[int, ...] | list[int] = _CQT_OCTAVES,
    cqt_bins_per_octave: tuple[int, ...] | list[int] = _CQT_BINS,
) -> dict[str, Any]:
    """Return the shared build config after the checks the three modules must share."""

    sampling_rate = _positive_int("sampling_rate", sampling_rate)
    if sampling_rate != _SAMPLE_RATE:
        raise BuildError(f"SEMamba++ sampling_rate must be {_SAMPLE_RATE}, got {sampling_rate}")
    n_fft = _positive_int("n_fft", n_fft)
    if n_fft != 400:
        raise BuildError(f"SEMamba++ n_fft must be 400, got {n_fft}")
    hop = _positive_int("hop", hop)
    win = _positive_int("win", win)
    if win != n_fft:
        raise BuildError(f"SEMamba++ win {win} must equal n_fft {n_fft}")
    if scan_backend not in {"reference", "fused"}:
        raise BuildError(f"unknown selective scan backend: {scan_backend!r}")
    input_channel = _positive_int("input_channel", input_channel)
    output_channel = _positive_int("output_channel", output_channel)
    if input_channel != 2 or output_channel != 1:
        raise BuildError("SEMamba++ channels must be input 2 and output 1")
    norm_epsilon = _positive_float("norm_epsilon", norm_epsilon)
    resolutions = _resolution_rows(mrd_resolutions)
    hops = _int_row("cqt_hops", cqt_hops)
    octaves = _int_row("cqt_octaves", cqt_octaves)
    bins = _int_row("cqt_bins_per_octave", cqt_bins_per_octave)
    if not hops or len(hops) != len(octaves) or len(hops) != len(bins):
        raise BuildError("CQT hop, octave, and bins lists must have the same non-zero length")
    return {
        "cqt_bins_per_octave": bins,
        "cqt_channels": _positive_int("cqt_channels", cqt_channels),
        "cqt_hops": hops,
        "cqt_octaves": octaves,
        "d_conv": _positive_int("d_conv", d_conv),
        "d_state": _positive_int("d_state", d_state),
        "expand": _positive_int("expand", expand),
        "hid_feature": _positive_int("hid_feature", hid_feature),
        "hop": hop,
        "input_channel": input_channel,
        "mrd_channels": _positive_int("mrd_channels", mrd_channels),
        "mrd_resolutions": resolutions,
        "n_fft": n_fft,
        "norm_epsilon": norm_epsilon,
        "num_tfmamba": _positive_int("num_tfmamba", num_tfmamba),
        "output_channel": output_channel,
        "sampling_rate": sampling_rate,
        "scan_backend": scan_backend,
        "win": win,
    }


def network_config(architecture: dict[str, Any]) -> dict[str, Any]:
    """Config passed to ``build_model``. STFT and discriminator fields are not network keys."""

    skipped = {"hop", "sampling_rate", "win"} | _DISCRIMINATOR_KEYS
    return {key: value for key, value in architecture.items() if key not in skipped}


def pack_forward(values: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the generator tuple onto the names used by ``SEMambaTask._generate``."""

    if len(values) != len(_FORWARD_FIELDS):
        raise RuntimeFailure(
            f"SEMamba++ generator returned {len(values)} tensors, expected {len(_FORWARD_FIELDS)}"
        )
    packed = dict(zip(_FORWARD_FIELDS, values, strict=True))
    for name, value in packed.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeFailure(f"SEMamba++ field {name} is not a tensor")
    return packed


def scan_execution(module: torch.nn.Module, device: torch.device) -> dict[str, str]:
    """Report the configured backend and the kernel this device will run.

    ``SelectiveSSM`` uses the fused kernel only when ``scan_backend`` is
    ``fused`` and the activation is CUDA. A fused setting on CPU therefore
    executes the reference scan. That is recorded here. It is not changed
    into a reference setting, and a missing CUDA extension is not treated as
    permission to run the reference kernel.
    """

    if not isinstance(device, torch.device):
        raise BuildError("scan execution needs a torch device")
    backends = [
        str(child.scan_backend)
        for child in module.modules()
        if child.__class__.__name__ == "SelectiveSSM" and hasattr(child, "scan_backend")
    ]
    if not backends:
        raise BuildError("SEMamba++ generator has no selective scan")
    configured = backends[0]
    if any(item != configured for item in backends):
        raise BuildError("SEMamba++ selective scan backends disagree inside one generator")
    if configured == "fused" and device.type == "cuda":
        try:
            import mamba_ssm.ops.selective_scan_interface  # noqa: F401
        except ImportError:
            executed = "unavailable"
        else:
            executed = "fused"
    elif configured == "fused":
        executed = "reference"
    else:
        executed = configured
    return {"configured": configured, "executed": executed}


class SEMambaPPGenerator(torch.nn.Module):
    """One SEMamba++ generator and the official STFT pair.

    ``forward`` is one generator evaluation. Accumulation of 1 keeps that
    graph. Accumulation above 1 calls this again after the discriminator step.
    """

    def __init__(
        self,
        *,
        seed: int,
        hid_feature: int = 48,
        num_tfmamba: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_epsilon: float = 1e-5,
        input_channel: int = 2,
        output_channel: int = 1,
        n_fft: int = 400,
        scan_backend: str = "reference",
        hop: int = 100,
        win: int = 400,
        sampling_rate: int = _SAMPLE_RATE,
        mrd_channels: int = 32,
        mrd_resolutions: tuple[tuple[int, int, int], ...] | list[list[int]] = _MRD_RESOLUTIONS,
        cqt_channels: int = 128,
        cqt_hops: tuple[int, ...] | list[int] = _CQT_HOPS,
        cqt_octaves: tuple[int, ...] | list[int] = _CQT_OCTAVES,
        cqt_bins_per_octave: tuple[int, ...] | list[int] = _CQT_BINS,
    ) -> None:
        super().__init__()
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            hid_feature=hid_feature,
            num_tfmamba=num_tfmamba,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            norm_epsilon=norm_epsilon,
            input_channel=input_channel,
            output_channel=output_channel,
            n_fft=n_fft,
            scan_backend=scan_backend,
            hop=hop,
            win=win,
            sampling_rate=sampling_rate,
            mrd_channels=mrd_channels,
            mrd_resolutions=mrd_resolutions,
            cqt_channels=cqt_channels,
            cqt_hops=cqt_hops,
            cqt_octaves=cqt_octaves,
            cqt_bins_per_octave=cqt_bins_per_octave,
        )
        with seed_model_init(self.seed):
            generator = _build_generator(self.architecture)
        self.generator = generator
        self.n_fft = int(self.architecture["n_fft"])
        self.hop = int(self.architecture["hop"])
        self.win = int(self.architecture["win"])
        self.compress = _COMPRESS
        self.sampling_rate = int(self.architecture["sampling_rate"])

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the official generation tuple for one micro-batch."""

        if not isinstance(clean, torch.Tensor) or not isinstance(noisy, torch.Tensor):
            raise RuntimeFailure("SEMamba++ inputs must be tensors")
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(
                f"expected matching [batch, time] waveforms, got {tuple(clean.shape)} and {tuple(getattr(noisy, 'shape', ()))}"
            )
        length = int(clean.shape[-1])
        try:
            clean_mag, clean_phase, clean_complex = semamba_mag_phase_stft(
                clean, self.n_fft, self.hop, self.win, self.compress, addeps=False
            )
            noisy_mag, noisy_phase, _noisy_complex = semamba_mag_phase_stft(
                noisy, self.n_fft, self.hop, self.win, self.compress, addeps=False
            )
            magnitude, phase = self.generator(noisy_mag, noisy_phase)
            if magnitude.shape != noisy_mag.shape:
                magnitude = _match_bins(magnitude, int(noisy_mag.shape[-2]))
                phase = _match_bins(phase, int(noisy_phase.shape[-2]))
            complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
            audio = semamba_mag_phase_istft(
                magnitude, phase, self.n_fft, self.hop, self.win, self.compress, length=length
            )
            if int(audio.shape[-1]) != length:
                audio = _fit_time(audio, length)
            _recomputed_mag, _recomputed_phase, recomputed = semamba_mag_phase_stft(
                audio, self.n_fft, self.hop, self.win, self.compress, addeps=True
            )
            del _recomputed_mag, _recomputed_phase
        except (TrainingError, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(str(exc)) from exc
        return (
            clean_mag,
            clean_phase,
            clean_complex,
            magnitude,
            phase,
            complex_spectrum,
            recomputed,
            audio,
        )

    def export_for_inference(self) -> dict[str, Any]:
        """Inference checkpoint. The value is the SEMambaPP state."""

        return {"generator": self.generator.state_dict()}


class SEMambaPPMRD(torch.nn.Module):
    """Multi-resolution discriminator. The constructor discards one generator."""

    def __init__(
        self,
        *,
        seed: int,
        hid_feature: int = 48,
        num_tfmamba: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_epsilon: float = 1e-5,
        input_channel: int = 2,
        output_channel: int = 1,
        n_fft: int = 400,
        scan_backend: str = "reference",
        hop: int = 100,
        win: int = 400,
        sampling_rate: int = _SAMPLE_RATE,
        mrd_channels: int = 32,
        mrd_resolutions: tuple[tuple[int, int, int], ...] | list[list[int]] = _MRD_RESOLUTIONS,
        cqt_channels: int = 128,
        cqt_hops: tuple[int, ...] | list[int] = _CQT_HOPS,
        cqt_octaves: tuple[int, ...] | list[int] = _CQT_OCTAVES,
        cqt_bins_per_octave: tuple[int, ...] | list[int] = _CQT_BINS,
    ) -> None:
        super().__init__()
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            hid_feature=hid_feature,
            num_tfmamba=num_tfmamba,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            norm_epsilon=norm_epsilon,
            input_channel=input_channel,
            output_channel=output_channel,
            n_fft=n_fft,
            scan_backend=scan_backend,
            hop=hop,
            win=win,
            sampling_rate=sampling_rate,
            mrd_channels=mrd_channels,
            mrd_resolutions=mrd_resolutions,
            cqt_channels=cqt_channels,
            cqt_hops=cqt_hops,
            cqt_octaves=cqt_octaves,
            cqt_bins_per_octave=cqt_bins_per_octave,
        )
        with seed_model_init(self.seed):
            _discard(_build_generator(self.architecture))
            discriminator = MultiResolutionDiscriminator(
                self.architecture["mrd_resolutions"],
                int(self.architecture["mrd_channels"]),
            )
        self.discriminator = discriminator

    def forward(
        self, waveform: torch.Tensor, paired: torch.Tensor | None = None
    ) -> tuple[Any, ...]:
        """Score real and, when given, fake waveforms."""

        return self.discriminator(waveform, paired)


class SEMambaPPCQTDiscriminator(torch.nn.Module):
    """CQT discriminator. The constructor discards one generator and one MRD."""

    def __init__(
        self,
        *,
        seed: int,
        hid_feature: int = 48,
        num_tfmamba: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_epsilon: float = 1e-5,
        input_channel: int = 2,
        output_channel: int = 1,
        n_fft: int = 400,
        scan_backend: str = "reference",
        hop: int = 100,
        win: int = 400,
        sampling_rate: int = _SAMPLE_RATE,
        mrd_channels: int = 32,
        mrd_resolutions: tuple[tuple[int, int, int], ...] | list[list[int]] = _MRD_RESOLUTIONS,
        cqt_channels: int = 128,
        cqt_hops: tuple[int, ...] | list[int] = _CQT_HOPS,
        cqt_octaves: tuple[int, ...] | list[int] = _CQT_OCTAVES,
        cqt_bins_per_octave: tuple[int, ...] | list[int] = _CQT_BINS,
    ) -> None:
        super().__init__()
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            hid_feature=hid_feature,
            num_tfmamba=num_tfmamba,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            norm_epsilon=norm_epsilon,
            input_channel=input_channel,
            output_channel=output_channel,
            n_fft=n_fft,
            scan_backend=scan_backend,
            hop=hop,
            win=win,
            sampling_rate=sampling_rate,
            mrd_channels=mrd_channels,
            mrd_resolutions=mrd_resolutions,
            cqt_channels=cqt_channels,
            cqt_hops=cqt_hops,
            cqt_octaves=cqt_octaves,
            cqt_bins_per_octave=cqt_bins_per_octave,
        )
        with seed_model_init(self.seed):
            _discard(_build_generator(self.architecture))
            _discard(
                MultiResolutionDiscriminator(
                    self.architecture["mrd_resolutions"],
                    int(self.architecture["mrd_channels"]),
                )
            )
            discriminator = MultiScaleSubbandCQTDiscriminator(
                int(self.architecture["sampling_rate"]),
                hops=self.architecture["cqt_hops"],
                octaves=self.architecture["cqt_octaves"],
                bins=self.architecture["cqt_bins_per_octave"],
                channels=int(self.architecture["cqt_channels"]),
            )
        self.discriminator = discriminator

    def forward(
        self, waveform: torch.Tensor, paired: torch.Tensor | None = None
    ) -> tuple[Any, ...]:
        """Score real and, when given, fake waveforms."""

        return self.discriminator(waveform, paired)


def _build_generator(architecture: dict[str, Any]) -> SEMambaPP:
    try:
        generator = build_model("semambapp", network_config(architecture), variant="semambapp")
    except (TrainingError, ValueError, RuntimeError) as exc:
        raise BuildError(str(exc)) from exc
    if not isinstance(generator, SEMambaPP):
        raise BuildError(f"semambapp builder returned {type(generator).__name__}, expected SEMambaPP")
    return generator


def _discard(module: torch.nn.Module) -> None:
    """Drop a module after its constructor has consumed the init stream."""

    del module


def _match_bins(value: torch.Tensor, bins: int) -> torch.Tensor:
    current = int(value.shape[-2])
    if current == bins:
        return value
    if current > bins:
        return value[..., :bins, :]
    return torch.nn.functional.pad(value, (0, 0, 0, bins - current))


def _fit_time(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if int(waveform.shape[-1]) > length:
        return waveform[..., :length]
    return torch.nn.functional.pad(waveform, (0, length - int(waveform.shape[-1])))


def _seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BuildError("seed must be an integer")
    return seed


def _resolution_rows(value: object) -> tuple[tuple[int, int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        raise BuildError("mrd_resolutions must be a non-empty list of [n_fft, hop, win]")
    rows = []
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, (list, tuple)) or len(item) != 3:
            raise BuildError("each MRD resolution must be [n_fft, hop, win]")
        rows.append(tuple(_positive_int("mrd resolution", part) for part in item))
    return tuple(rows)


def _int_row(name: str, value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        raise BuildError(f"{name} must be a non-empty list of positive integers")
    return tuple(_positive_int(name, item) for item in value)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{name} must be a positive number")
    number = float(value)
    if number <= 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise BuildError(f"{name} must be a positive finite number")
    return number
