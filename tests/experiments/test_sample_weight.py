"""Tail batches stay in the epoch and the loss uses the real sample count."""

from __future__ import annotations

import unittest

import torch

from src.experiments.data.sample import SampleRef
from src.experiments.data.fixed_pair import MicroBatcher
from src.experiments.data.tensor import TensorPairData
from src.experiments.data.sample import Batch
from src.experiments.execution.runtime import PreparePolicy, PreparedModel, ProcessRuntime
from src.experiments.official.tf_gridnet.trainer import GridNetUpdate
from src.experiments.training.control import (
    open_flat_window_at,
    open_grouped_window,
    window_divisor,
    window_scale,
)
from src.experiments.training.objectives import ObjectiveOutput
from src.experiments.training.sample_weight import sample_scales, window_is_uniform


def _ref(index: int) -> SampleRef:
    return SampleRef(sample_id=str(index), realization_id=str(index), index=index)


class ScaleTests(unittest.TestCase):
    def test_uniform_window_matches_divide_by_accumulation(self) -> None:
        counts = [[4, 4], [4, 4]]
        self.assertTrue(window_is_uniform(counts, 2))
        self.assertFalse(window_is_uniform([[4, 4], [4, 1]], 2))
        self.assertFalse(window_is_uniform([[4]], 2))

    def test_uneven_scales_reconstruct_the_global_mean(self) -> None:
        local = [4.0, 1.0]
        counts = [4, 1]
        scales = sample_scales(counts, world_size=2, n_global=7)
        other = sample_scales([2], world_size=2, n_global=7)
        weighted = sum(loss * scale for loss, scale in zip(local, scales))
        other_weighted = 9.0 * other[0]
        self.assertAlmostEqual(weighted + other_weighted, (4 * 4 + 1 * 1 + 2 * 9) / 7 * 2)
        self.assertEqual(sample_scales([0], world_size=2, n_global=5), [0.0])
        self.assertEqual(sample_scales([3], world_size=1, n_global=0), [0.0])

    def test_equal_rows_keep_the_integer_divisor(self) -> None:
        accumulation = 2
        scale = window_scale([4, 4], [[4, 4]], accumulation=accumulation, world_size=1)
        self.assertIs(scale, accumulation)
        short = window_scale([4], [[4]], accumulation=accumulation, world_size=1)
        self.assertEqual(short, 1)
        mixed = window_scale([2, 1], [[2, 1]], accumulation=accumulation, world_size=1)
        self.assertEqual(mixed, [2 / 3, 1 / 3])


class TailBatchTests(unittest.TestCase):
    def test_keep_partial_keeps_the_short_group_and_records_drops(self) -> None:
        rows = [_ref(index) for index in range(5)]
        kept = MicroBatcher(2, 2)
        self.assertTrue(kept.keep_partial)
        self.assertEqual([len(group) for group in kept.groups(rows)], [2, 2, 1])
        self.assertEqual(kept.dropped_samples, 0)
        dropped = MicroBatcher(2, 2, keep_partial=False)
        self.assertEqual([len(group) for group in dropped.groups(rows)], [2, 2])
        self.assertEqual(dropped.dropped_samples, 1)

    def test_tensor_pairs_keep_the_tail_without_copying(self) -> None:
        data = TensorPairData(features=3, examples=5, batch_size=2, seed=1)
        data.setup(type("Runtime", (), {"device": torch.device("cpu"), "world_size": 1, "rank": 0})())
        view = data.train_epoch(0, start_update=0)
        self.assertEqual(view.updates, 3)
        self.assertEqual(view.batch(0)[0].inputs.shape[0], 2)
        self.assertEqual(view.batch(2)[0].inputs.shape[0], 1)
        self.assertEqual(view.batch(2)[0].sample_id, "ex-4")
        explicit = TensorPairData(features=3, examples=5, batch_size=2, seed=1, drop_last=True)
        explicit.setup(type("Runtime", (), {"device": torch.device("cpu"), "world_size": 1, "rank": 0})())
        self.assertEqual(explicit.train_epoch(0, start_update=0).updates, 2)

    def test_short_rank_gets_no_copied_sample(self) -> None:
        data = TensorPairData(features=3, examples=5, batch_size=2, seed=1)
        runtime = type("Runtime", (), {"device": torch.device("cpu"), "world_size": 2, "rank": 1})()
        data.setup(runtime)
        view = data.train_epoch(0, start_update=0)
        self.assertEqual(view.updates, 2)
        self.assertEqual(view.batch(0)[0].inputs.shape[0], 2)
        self.assertEqual(view.batch(1), [])


class OfficialWindowTests(unittest.TestCase):
    def test_a_short_tail_is_kept_and_a_full_window_still_uses_accumulation(self) -> None:
        accumulation = 2
        self.assertIs(window_divisor(accumulation, accumulation), accumulation)
        self.assertEqual(window_divisor(accumulation, 1), 1)
        data = TensorPairData(features=2, examples=5, batch_size=1, seed=1)
        data.setup(type("Runtime", (), {"device": torch.device("cpu"), "world_size": 1, "rank": 0})())
        self.assertEqual(data.train_epoch(0, start_update=0).updates, 5)
        holder = type("Trainer", (), {"data": data, "accumulation": accumulation})()
        _, index, _, planned = open_grouped_window(holder, "trainer")
        self.assertEqual(index, 0)
        self.assertEqual(len(planned), accumulation)
        data.update_index = 4
        _, index, _, planned = open_grouped_window(holder, "trainer")
        self.assertEqual((index, len(planned)), (4, 1))
        self.assertEqual(planned[0][0].inputs.shape[0], 1)
        data.update_index = 0
        _, _, _, batches = open_flat_window_at(data, accumulation, "trainer")
        self.assertEqual(len(batches), accumulation)
        data.update_index = 4
        _, _, _, batches = open_flat_window_at(data, accumulation, "trainer")
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].sample_id, "ex-4")


class _Rows(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if int(inputs.shape[0]) == 0:
            raise AssertionError("empty micro-batch was forwarded")
        return inputs * self.weight


class _Mean(torch.nn.Module):
    def forward(self, context: dict[str, torch.Tensor]) -> ObjectiveOutput:
        value = torch.nn.functional.mse_loss(context["estimate"], context["target"])
        return ObjectiveOutput(value, (("mse", value, 1.0),))


def _batch(rows: int, sample_id: str) -> Batch:
    generator = torch.Generator().manual_seed(rows + len(sample_id))
    inputs = torch.randn(rows, 2, generator=generator)
    return Batch(inputs=inputs, targets=inputs + 0.1, sample_id=sample_id, realization_id=sample_id)


class OfficialScaleTests(unittest.TestCase):
    def _update(self) -> tuple[GridNetUpdate, ProcessRuntime, torch.nn.Parameter]:
        module = _Rows()
        prepared = PreparedModel(
            component_id="model",
            module=module,
            parameters=tuple(module.parameters()),
            state_module=module,
            policy=PreparePolicy(),
        )
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        update = GridNetUpdate(
            prepared, _Mean(), torch.optim.SGD(module.parameters(), lr=0.1), accumulation=2, grad_clip=1.0,
        )
        return update, runtime, module.weight

    def test_mixed_rows_match_the_sample_mean_and_an_empty_row_is_not_forwarded(self) -> None:
        update, runtime, weight = self._update()
        wide = _batch(2, "wide")
        narrow = _batch(1, "narrow")
        empty = Batch(
            inputs=torch.empty(0, 2),
            targets=torch.empty(0, 2),
            sample_id="empty",
            realization_id="empty",
        )
        update.run([[wide], [narrow]], runtime, "trainer", apply_optimizer=False)
        mixed = weight.grad.detach().clone()
        weight.grad = None
        update.run([[wide], [empty]], runtime, "trainer", apply_optimizer=False)
        kept = weight.grad.detach().clone()

        def _mean_grad(batch: Batch) -> torch.Tensor:
            alone = _Rows()
            alone.weight.data.copy_(weight.data)
            loss = torch.nn.functional.mse_loss(alone(batch.inputs), batch.targets)
            loss.backward()
            assert alone.weight.grad is not None
            return alone.weight.grad.detach().clone()

        expected = _mean_grad(wide) * (2 / 3) + _mean_grad(narrow) * (1 / 3)
        self.assertLess(torch.max(torch.abs(mixed - expected)).item(), 1e-6)
        self.assertLess(torch.max(torch.abs(kept - _mean_grad(wide))).item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
