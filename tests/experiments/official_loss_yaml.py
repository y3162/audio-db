"""Loss-graph fragments for experiment YAML assembled inside tests.

The coefficients match the published recipes. Production fixtures carry the
same graph; these strings exist so inline templates build the trainers.
"""

MP_COMPONENTS = """\
  magnitude_loss:
    role: loss
    class: src.experiments.losses.MagnitudeMSE
    kwargs:
      name: magnitude
      estimate: generated_mag
      target: clean_mag
  phase_loss:
    role: loss
    class: src.experiments.losses.PhaseLoss
    kwargs:
      name: phase
      mode: diff
  complex_loss:
    role: loss
    class: src.experiments.losses.ScaledComplexMSE
    kwargs:
      name: complex
      scale: 2.0
      estimate: generated_complex
      target: clean_complex
  consistency_loss:
    role: loss
    class: src.experiments.losses.ScaledComplexMSE
    kwargs:
      name: consistency
      scale: 2.0
      estimate: recomputed_complex
      target: generated_complex
  time_loss:
    role: loss
    class: src.experiments.losses.TimeL1
    kwargs:
      name: time
      estimate: generated_audio
      target: clean_audio
  metric_loss:
    role: loss
    class: src.experiments.losses.MetricMSE
    kwargs:
      name: metric
  signal_objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        magnitude: 0.9
        phase: 0.3
        complex: 0.1
        consistency: 0.1
        time: 0.2
    inject:
      terms:
        - {ref: magnitude_loss}
        - {ref: phase_loss}
        - {ref: complex_loss}
        - {ref: consistency_loss}
        - {ref: time_loss}
  metric_objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        metric: 0.05
    inject:
      terms:
        - {ref: metric_loss}
  discriminator_objective:
    role: loss
    class: src.experiments.losses.MPSENetDiscriminatorObjective
"""

MP_INJECT = """\
      signal_objective: {ref: signal_objective}
      metric_objective: {ref: metric_objective}
      discriminator_objective: {ref: discriminator_objective}
"""

MOSS_COMPONENTS = """\
  ri_loss:
    role: loss
    class: src.experiments.losses.ContextTerm
    kwargs:
      name: ri
  magnitude_loss:
    role: loss
    class: src.experiments.losses.ContextTerm
    kwargs:
      name: magnitude
  time_loss:
    role: loss
    class: src.experiments.losses.ContextTerm
    kwargs:
      name: time
  gan_loss:
    role: loss
    class: src.experiments.losses.ContextTerm
    kwargs:
      name: gan
  generator_objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        ri: 0.1
        magnitude: 0.9
        time: 0.2
        gan: 0.05
    inject:
      terms:
        - {ref: ri_loss}
        - {ref: magnitude_loss}
        - {ref: time_loss}
        - {ref: gan_loss}
  discriminator_objective:
    role: loss
    class: src.experiments.losses.MossFormerDiscriminatorObjective
"""

MOSS_INJECT = """\
      generator_objective: {ref: generator_objective}
      discriminator_objective: {ref: discriminator_objective}
"""

DFN_COMPONENTS = """\
  spectral_loss:
    role: loss
    class: src.experiments.official.deepfilternet3.losses.DeepFilterNetSpectralLoss
    kwargs:
      name: spectral
      factor: 500.0
      complex_factor: 500.0
      gamma: 0.3
      fft_sizes: [256, 512, 1024, 2048]
      magnitude_floor: 1.0e-12
  local_snr_loss:
    role: loss
    class: src.experiments.official.deepfilternet3.losses.LocalSNRLoss
    kwargs:
      name: local_snr
      milliseconds: 20
      energy_floor: 1.0e-12
  objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        spectral: 1.0
        local_snr: 0.001
    inject:
      terms:
        - {ref: spectral_loss}
        - {ref: local_snr_loss}
"""

DFN_INJECT = """\
      objective: {ref: objective}
"""

SPECTRAL_OBJECTIVE = """\
  objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        mr_l1: 1.0
    inject:
      terms:
        - {ref: loss}
"""

SEMAMBA_COMPONENTS = """\
  adversarial_loss:
    role: loss
    class: src.experiments.losses.SummedLeastSquaresGenerator
    kwargs:
      name: adversarial
      first: fake_cqt
      second: fake_mrd
  matching_loss:
    role: loss
    class: src.experiments.losses.PairedFeatureMatching
    kwargs:
      name: matching
      scale: 2.0
      reduction: sum
      first_real: real_cqt_features
      first_fake: fake_cqt_features
      second_real: real_mrd_features
      second_fake: fake_mrd_features
  mel_loss:
    role: loss
    class: src.experiments.losses.SlaneyLogMelLoss
    kwargs:
      name: mel
      n_mels: [40, 80, 160, 320]
      windows: [256, 512, 1024, 2048]
      clamp: 1.0e-5
  magnitude_loss:
    role: loss
    class: src.experiments.losses.MagnitudeMSE
    kwargs:
      name: magnitude
      estimate: generated_mag
      target: clean_mag
  phase_loss:
    role: loss
    class: src.experiments.losses.PhaseLoss
    kwargs:
      name: phase
      mode: matrix
  complex_loss:
    role: loss
    class: src.experiments.losses.ScaledComplexMSE
    kwargs:
      name: complex
      scale: 2.0
      estimate: generated_complex
      target: clean_complex
  consistency_loss:
    role: loss
    class: src.experiments.losses.ScaledComplexMSE
    kwargs:
      name: consistency
      scale: 2.0
      estimate: recomputed_complex
      target: generated_complex
  generator_objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        adversarial: 1.0
        matching: 1.0
        mel: 0.1
        magnitude: 0.9
        phase: 0.3
        complex: 0.1
        consistency: 0.1
    inject:
      terms:
        - {ref: adversarial_loss}
        - {ref: matching_loss}
        - {ref: mel_loss}
        - {ref: magnitude_loss}
        - {ref: phase_loss}
        - {ref: complex_loss}
        - {ref: consistency_loss}
  discriminator_term:
    role: loss
    class: src.experiments.losses.SummedLeastSquaresDiscriminator
    kwargs:
      name: discriminator
      first_real: real_cqt
      first_fake: fake_cqt
      second_real: real_mrd
      second_fake: fake_mrd
  discriminator_objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        discriminator: 1.0
    inject:
      terms:
        - {ref: discriminator_term}
"""

SEMAMBA_INJECT = """\
      generator_objective: {ref: generator_objective}
      discriminator_objective: {ref: discriminator_objective}
"""
