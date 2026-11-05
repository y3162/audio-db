"""Two-rank real SE Trainer and TensorBoard smoke worker."""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.core.batch import plan_batches
from src.training.core.distributed import setup_process, unwrap
from src.training.core.trainer import Trainer
from src.training.se.mp_senet import MPSENetTask
from tests.training.support import paired_dataset, tiny_config
from tests.training.test_model_tasks import _base
from tests.training.test_tensorboard_integration import model_cases


def case(name, root):
    if name in ('mp_senet', 'mp_senet_official'):
        length = 1024 if name == 'mp_senet_official' else 512
        config = tiny_config(root / 'ckpt', max_steps=2, seed=7, segment_size=length,
                             global_batch_size=2, micro_batch_size_per_gpu=1)
        if name == 'mp_senet_official':
            config['loss'] = 'official'
            config['model'].update(n_fft=64, hop_size=16, win_size=64)
        return MPSENetTask, config, length
    cls, params, length = model_cases()[name]
    config = _base(root, name, segment_size=length, **params)
    config.update(global_batch_size=2, micro_batch_size_per_gpu=1, max_steps=2)
    return cls, config, length


def main():
    rank, port, root, name = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=port, RANK=str(rank),
                      WORLD_SIZE='2', LOCAL_RANK=str(rank))
    process = setup_process(os.environ.get('AUDIO_DB_TB_DDP_DEVICE', 'cpu'))
    cls, config, length = case(name, root)
    global_batch = int(os.environ.get('AUDIO_DB_TB_DDP_GLOBAL_BATCH', '2'))
    config['global_batch_size'] = global_batch
    config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
        'audio': {'enabled': True, 'max_cases': 1, 'max_seconds': 1}, 'spectrogram': True}}
    train = paired_dataset(root / 'train', 2 * global_batch, frames=length, segment_size=length, seed=7, rewrite=False)
    valid = paired_dataset(root / 'valid', 1, frames=length, segment_size=length, seed=8, rewrite=False)
    task = cls(config, process.device)
    trainer = Trainer(task, plan_batches(global_batch, 1, 2), process, max_steps=2, seed=7,
                      checkpoint_dir=root / 'ckpt', checkpoint_interval=0, validation_interval=2, config=config)
    try:
        trainer.fit(train, valid)
        modules = {key: value for key, value in vars(task).items()
                   if key == 'generator' or key.startswith('discriminator')}
        parameters = {f'{name}.{key}': parameter.detach().cpu()
                      for name, module in modules.items() if isinstance(module, torch.nn.Module)
                      for key, parameter in unwrap(module).named_parameters()}
        torch.save(parameters, root / f'parameters-{rank}.pt')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
