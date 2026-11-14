"""Official objective graphs and the checks that refuse a different one.

The numbers here are the published coefficients. Training reads the injected
module. These constants are only the lock for ``official_faithful`` and
``leakage_safe``, and the explicit graph used by tests that build a trainer
without YAML.
"""

from __future__ import annotations

from src.experiments.errors import BuildError
from src.experiments.training.loss_terms import (
    ContextTerm,
    MagnitudeMSE,
    MetricMSE,
    MossFormerDiscriminatorObjective,
    MPSENetDiscriminatorObjective,
    PairedFeatureMatching,
    PhaseLoss,
    ScaledComplexMSE,
    SlaneyLogMelLoss,
    SummedLeastSquaresDiscriminator,
    SummedLeastSquaresGenerator,
    TimeL1,
)
from src.experiments.training.objectives import WeightedObjective

_LOCKED = frozenset({"official_faithful", "leakage_safe"})

# name, weight, class, constructor kwargs besides name
MP_SIGNAL = (
    ("magnitude", 0.9, MagnitudeMSE, {"estimate": "generated_mag", "target": "clean_mag"}),
    ("phase", 0.3, PhaseLoss, {"mode": "diff"}),
    ("complex", 0.1, ScaledComplexMSE, {"scale": 2.0, "estimate": "generated_complex", "target": "clean_complex"}),
    ("consistency", 0.1, ScaledComplexMSE, {"scale": 2.0, "estimate": "recomputed_complex", "target": "generated_complex"}),
    ("time", 0.2, TimeL1, {"estimate": "generated_audio", "target": "clean_audio"}),
)
MP_METRIC = (("metric", 0.05, MetricMSE, {}),)
MOSS_GENERATOR = (
    ("ri", 0.1, ContextTerm, {"key": "ri"}),
    ("magnitude", 0.9, ContextTerm, {"key": "magnitude"}),
    ("time", 0.2, ContextTerm, {"key": "time"}),
    ("gan", 0.05, ContextTerm, {"key": "gan"}),
)
SEMAMBA_GENERATOR = (
    ("adversarial", 1.0, SummedLeastSquaresGenerator, {"first": "fake_cqt", "second": "fake_mrd"}),
    (
        "matching",
        1.0,
        PairedFeatureMatching,
        {
            "scale": 2.0,
            "reduction": "sum",
            "first_real": "real_cqt_features",
            "first_fake": "fake_cqt_features",
            "second_real": "real_mrd_features",
            "second_fake": "fake_mrd_features",
        },
    ),
    (
        "mel",
        0.1,
        SlaneyLogMelLoss,
        {"n_mels": (40, 80, 160, 320), "windows": (256, 512, 1024, 2048), "clamp": 1e-5},
    ),
    ("magnitude", 0.9, MagnitudeMSE, {"estimate": "generated_mag", "target": "clean_mag"}),
    ("phase", 0.3, PhaseLoss, {"mode": "matrix"}),
    ("complex", 0.1, ScaledComplexMSE, {"scale": 2.0, "estimate": "generated_complex", "target": "clean_complex"}),
    (
        "consistency",
        0.1,
        ScaledComplexMSE,
        {"scale": 2.0, "estimate": "recomputed_complex", "target": "generated_complex"},
    ),
)
SEMAMBA_DISCRIMINATOR = {
    "first_real": "real_cqt",
    "first_fake": "fake_cqt",
    "second_real": "real_mrd",
    "second_fake": "fake_mrd",
}


def weighted_objective(spec: tuple[tuple[str, float, type, dict], ...]) -> WeightedObjective:
    terms = [cls(name, **kwargs) for name, _weight, cls, kwargs in spec]
    weights = {name: weight for name, weight, _cls, _kwargs in spec}
    return WeightedObjective(terms, weights)


def mp_senet_signal_objective() -> WeightedObjective:
    return weighted_objective(MP_SIGNAL)


def mp_senet_metric_objective() -> WeightedObjective:
    return weighted_objective(MP_METRIC)


def mp_senet_discriminator_objective() -> MPSENetDiscriminatorObjective:
    return MPSENetDiscriminatorObjective()


def mossformer_generator_objective() -> WeightedObjective:
    return weighted_objective(MOSS_GENERATOR)


def mossformer_discriminator_objective() -> MossFormerDiscriminatorObjective:
    return MossFormerDiscriminatorObjective()


def semamba_generator_objective() -> WeightedObjective:
    return weighted_objective(SEMAMBA_GENERATOR)


def semamba_discriminator_objective() -> WeightedObjective:
    term = SummedLeastSquaresDiscriminator("discriminator", **SEMAMBA_DISCRIMINATOR)
    return WeightedObjective([term], {"discriminator": 1.0})


def require_weighted(objective: object, claim: str, spec: tuple[tuple[str, float, type, dict], ...], *, what: str) -> None:
    if claim not in _LOCKED:
        return
    if type(objective) is not WeightedObjective:
        raise BuildError(f"{claim} {what} must be WeightedObjective")
    if objective.check_finite:
        raise BuildError(f"{claim} {what} cannot set check_finite")
    expected = tuple(name for name, _weight, _cls, _kwargs in spec)
    if objective.order != expected:
        raise BuildError(f"{claim} {what} term order must be {expected}")
    for name, weight, cls, kwargs in spec:
        if objective.weights[name] != weight:
            raise BuildError(f"{claim} {what} weight for {name} must be {weight}")
        term = objective.term(name)
        if type(term) is not cls:
            raise BuildError(f"{claim} {what} term {name} must be {cls.__name__}")
        for key, value in kwargs.items():
            got = getattr(term, key)
            if got != value:
                raise BuildError(f"{claim} {what} term {name} {key} must be {value}")


def require_objective_type(objective: object, claim: str, cls: type, *, what: str) -> None:
    if claim not in _LOCKED:
        return
    if type(objective) is not cls:
        raise BuildError(f"{claim} {what} must be {cls.__name__}")


def require_mp_senet(signal: object, metric: object, discriminator: object, claim: str) -> None:
    require_weighted(signal, claim, MP_SIGNAL, what="signal objective")
    require_weighted(metric, claim, MP_METRIC, what="metric objective")
    require_objective_type(discriminator, claim, MPSENetDiscriminatorObjective, what="discriminator objective")


def require_mossformer(generator: object, discriminator: object, claim: str) -> None:
    require_weighted(generator, claim, MOSS_GENERATOR, what="generator objective")
    require_objective_type(discriminator, claim, MossFormerDiscriminatorObjective, what="discriminator objective")
