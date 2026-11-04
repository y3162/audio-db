import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.core.checkpoint import load_checkpoint
from tests.training.support import cpu_process


class Dataset:
    name = 'fake'
    def __len__(self): return 3
    def identity(self): return {'name': self.name}
    def set_epoch(self, epoch): self.epoch = epoch
    def collate(self, indices): return {'case_id': [str(i) for i in indices]}


class Task:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.Adam([self.weight], lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)
        self.seen = []
    def distribute(self, process): pass
    def train_update(self, group, accumulation):
        self.seen.extend(group[0]['case_id'])
        self.optimizer.zero_grad()
        (self.weight - 1).square().backward()
        self.optimizer.step()
        return {'generator': float(self.weight.detach())}
    def step_schedulers(self, summary): self.scheduler.step(summary['losses']['generator']['value'])
    def state_dict(self): return {'weight': self.weight.detach().clone(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()}
    def load_state_dict(self, state):
        self.weight.data.copy_(state['weight'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
    def generator_state_dict(self): return {'weight': self.weight.detach().clone()}
    def eval_mode(self): pass
    def train_mode(self): pass
    def validation_report(self, batch):
        metric = {'status': 'ok', 'value': 1.0, 'reason': None}
        return {'losses': {'generator': metric}, 'pesq': metric, 'stoi': metric}
    def learning_rates(self): return {'generator': self.optimizer.param_groups[0]['lr']}


def run(path, steps, resume=None):
    config = {'seed': 1, 'loss': 'official', 'model': {'name': 'tf_gridnet'}, 'logical_steps_per_epoch': 4, 'best_checkpoint_start_epoch': 0}
    task = Task()
    trainer = Trainer(task, plan_batches(1, 1, 1), cpu_process(), max_steps=steps, seed=1, checkpoint_dir=path, checkpoint_interval=2, validation_interval=0, config=config)
    trainer.fit(Dataset(), Dataset(), resume=resume)
    return task


def test_logical_boundaries_and_resume(tmp_path):
    full = tmp_path / 'full'
    expected = run(full, 12)
    records = [json.loads(x) for x in (full / 'metrics.jsonl').read_text().splitlines()]
    assert [r['global_step'] for r in records if r['type'] == 'validation'] == [4, 8, 12]
    assert (full / 'best_generator.pt').exists()
    assert expected.scheduler.last_epoch == 3
    for step in (2, 4):
        dest = tmp_path / str(step)
        dest.mkdir()
        resumed = run(dest, 12, full / f'step_{step:08d}.pt')
        assert resumed.seen == expected.seen[step:]
        assert torch.equal(resumed.weight, expected.weight)
        assert resumed.scheduler.state_dict() == expected.scheduler.state_dict()
        payload = load_checkpoint(dest / 'step_00000012.pt', torch.device('cpu'))
        assert payload['logical_epoch'] == 3
        assert payload['best_loss'] == 1.0
        assert set(payload['rng']) >= {'python', 'numpy', 'torch'}


class LogicalEpochTests(unittest.TestCase):
    def test_resume_and_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            test_logical_boundaries_and_resume(Path(directory))
