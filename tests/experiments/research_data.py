"""Scripted data and a fixed recognizer for research-evaluation tests."""

from tests.experiments.tf_gridnet_data import ScriptedChunkData

TRANSCRIPT = "alpha bravo charlie"


class TranscriptChunkData(ScriptedChunkData):
    """Scripted waveforms plus a reference transcript and a train-eval split.

    ``train_eval_epoch`` yields the validation waveforms and does not assign
    ``epoch`` or ``update_index``.
    """

    def reference_transcript(self, sample_id: str) -> str:
        del sample_id
        return TRANSCRIPT

    def train_eval_epoch(self, epoch: int):
        return self.validation_epoch(epoch)


class FixedRecognizer:
    """Return one hypothesis for every utterance."""

    def __init__(self, hypothesis: str = "alpha bravo") -> None:
        self.hypothesis = hypothesis

    def transcribe(self, audio: object, sample_rate: int) -> str:
        del audio, sample_rate
        return self.hypothesis
