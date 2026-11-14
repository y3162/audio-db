"""Runtime surface used by trainers: forward, device move, and gradient sync."""

from __future__ import annotations

import unittest

import torch

from src.experiments.data.sample import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.execution.runtime import PreparePolicy, ProcessRuntime


class RuntimeApiTests(unittest.TestCase):
    def test_prepared_model_is_callable_and_keeps_parameter_identity(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        self.addCleanup(runtime.shutdown)
        module = torch.nn.Linear(3, 2, bias=False)
        prepared = runtime.prepare(module, PreparePolicy(trainable=True, distribute="none"))
        before = {id(parameter) for parameter in prepared.parameters}
        inputs = torch.ones(1, 3)
        direct = prepared.module(inputs)
        called = prepared(inputs)
        self.assertTrue(torch.equal(direct, called))
        self.assertEqual(before, {id(parameter) for parameter in prepared.parameters})
        self.assertEqual(before, {id(parameter) for parameter in prepared.module.parameters()})
        self.assertIs(prepared.module, prepared.state_module)

    def test_move_keeps_batch_metadata_and_tensor_identity_on_device(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        self.addCleanup(runtime.shutdown)
        inputs = torch.zeros(2, 4)
        targets = torch.ones(2, 4)
        lengths = torch.tensor([4, 3])
        batch = Batch(
            inputs,
            targets,
            "sample-a",
            "realization-a",
            lengths=lengths,
            sample_ids=("sample-a", "sample-b"),
            realization_ids=("realization-a", "realization-b"),
            teacher_kind="dry_clean",
            sample_rate=16000,
        )
        moved = runtime.move(batch)
        self.assertIs(moved, batch)
        self.assertEqual(moved.sample_id, "sample-a")
        self.assertEqual(moved.sample_ids, ("sample-a", "sample-b"))
        self.assertEqual(moved.teacher_kind, "dry_clean")
        self.assertEqual(moved.sample_rate, 16000)
        payload = {"wave": inputs, "name": "kept", "count": 2}
        self.assertIs(runtime.move(payload), payload)
        shifted = runtime.move({"wave": torch.zeros(1), "name": "kept"})
        self.assertEqual(shifted["name"], "kept")
        self.assertIsInstance(shifted["wave"], torch.Tensor)

    def test_move_refuses_modules(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        self.addCleanup(runtime.shutdown)
        with self.assertRaisesRegex(RuntimeFailure, "does not relocate modules"):
            runtime.move(torch.nn.Linear(2, 2))

    def test_sync_gradients_brackets_forward_and_backward_on_cpu(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        self.addCleanup(runtime.shutdown)
        module = torch.nn.Linear(2, 1, bias=False)
        prepared = runtime.prepare_model(module, PreparePolicy())
        optimizer = torch.optim.SGD(prepared.parameters, lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.ones(1, 2)
        with runtime.sync_gradients([prepared], enabled=False):
            loss = prepared(inputs).sum()
            runtime.backward(loss)
        self.assertIsNotNone(prepared.parameters[0].grad)
        result = runtime.step(optimizer)
        runtime.end_step()
        self.assertTrue(result.applied)
        with self.assertRaisesRegex(RuntimeFailure, "enabled must be a bool"):
            with runtime.sync_gradients([prepared], enabled="no"):  # type: ignore[arg-type]
                pass
        with runtime.sync_context([prepared], sync=True):
            again = prepared(torch.ones(1, 2)).sum()
            runtime.backward(again)
        runtime.step(optimizer)
        runtime.end_step()


if __name__ == "__main__":
    unittest.main()
