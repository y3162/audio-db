"""Official objective graphs and the published-coefficient comparison.

The numbers here are the published coefficients. Training reads the injected
module. ``official_faithful`` and ``leakage_safe`` record a difference when the
built objective does not match these constants. They do not refuse the run.
``research`` is not compared. Tests that build a trainer without YAML use the
same graphs.
"""

from __future__ import annotations

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


def require_weighted(objective: object, claim: str, spec: tuple[tuple[str, float, type, dict], ...], *, what: str) -> tuple[str, ...]:
    """Record coefficient and term differences. Do not refuse a changed loss.

    A missing attribute on a term that is present is still reported. The
    objective keeps running with the weights it was given.
    """

    if claim not in _LOCKED:
        return ()
    if type(objective) is not WeightedObjective:
        return (f"{claim} {what} is {type(objective).__name__}, published objective is WeightedObjective",)
    differences: list[str] = []
    if objective.check_finite:
        differences.append(f"{claim} {what} sets check_finite")
    expected = tuple(name for name, _weight, _cls, _kwargs in spec)
    if objective.order != expected:
        differences.append(f"{claim} {what} term order is {objective.order}, published order is {expected}")
    for name, weight, cls, kwargs in spec:
        if name not in objective.weights:
            differences.append(f"{claim} {what} is missing term {name}")
            continue
        if objective.weights[name] != weight:
            differences.append(
                f"{claim} {what} weight for {name} is {objective.weights[name]}, published weight is {weight}"
            )
        term = objective.term(name)
        if type(term) is not cls:
            differences.append(f"{claim} {what} term {name} is {type(term).__name__}, published class is {cls.__name__}")
        for key, value in kwargs.items():
            got = getattr(term, key)
            if got != value:
                differences.append(f"{claim} {what} term {name} {key} is {got!r}, published value is {value!r}")
    for name in objective.order:
        if name not in expected:
            differences.append(f"{claim} {what} adds term {name}")
    return tuple(differences)


def require_objective_type(objective: object, claim: str, cls: type, *, what: str) -> tuple[str, ...]:
    if claim not in _LOCKED:
        return ()
    if type(objective) is not cls:
        return (f"{claim} {what} is {type(objective).__name__}, published class is {cls.__name__}",)
    return ()


def require_mp_senet(signal: object, metric: object, discriminator: object, claim: str) -> tuple[str, ...]:
    return (
        *require_weighted(signal, claim, MP_SIGNAL, what="signal objective"),
        *require_weighted(metric, claim, MP_METRIC, what="metric objective"),
        *require_objective_type(discriminator, claim, MPSENetDiscriminatorObjective, what="discriminator objective"),
    )


def require_mossformer(generator: object, discriminator: object, claim: str) -> tuple[str, ...]:
    return (
        *require_weighted(generator, claim, MOSS_GENERATOR, what="generator objective"),
        *require_objective_type(discriminator, claim, MossFormerDiscriminatorObjective, what="discriminator objective"),
    )
