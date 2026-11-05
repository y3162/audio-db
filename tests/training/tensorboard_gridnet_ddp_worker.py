"""Two-rank TF-GridNet Trainer and TensorBoard smoke worker."""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.core.batch import plan_batches
from src.training.core.distributed import setup_process
from src.training.core.trainer import Trainer
from src.training.se.tf_gridnet import TFGridNetTask
from tests.training.support import paired_dataset
from tests.training.test_model_tasks import _base


def main():
    rank, port, root = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=port, RANK=str(rank),
                      WORLD_SIZE='2', LOCAL_RANK=str(rank))
    process = setup_process(os.environ.get('AUDIO_DB_TB_DDP_DEVICE', 'cpu'))
    architecture = dict(n_fft=32, stride=16, n_layers=1, lstm_hidden_units=8, attn_n_head=1,
                        attn_approx_qk_dim=8, emb_dim=8, emb_ks=1, emb_hs=1)
    config = _base(root, 'tf_gridnet', variant='ms-snsd', architecture=architecture, segment_size=800)
    config['logging'] = {'tensorboard': {'enabled': True, 'scalar_interval': 1,
        'audio': {'enabled': True, 'max_cases': 1, 'max_seconds': 1}, 'spectrogram': True}}
    train = paired_dataset(root / 'train', 4, frames=800, segment_size=800, seed=7, rewrite=False)
    valid = paired_dataset(root / 'valid', 1, frames=800, segment_size=800, seed=8, rewrite=False)
    steps = int(os.environ.get('AUDIO_DB_TB_GRID_STEPS', '1'))
    task = TFGridNetTask(config, process.device)
    trainer = Trainer(task, plan_batches(4, 1, 2), process, max_steps=steps, seed=7,
                      checkpoint_dir=root / 'ckpt', checkpoint_interval=0, validation_interval=steps, config=config)
    if rank == 0:
        backend = dist.get_backend() if dist.is_initialized() else 'none'
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if process.uses_cuda else []
        (root / 'ddp-backend.txt').write_text(
            f'backend={backend}\nworld={process.world_size}\ndevice={process.device}\n'
            f'loss={config.get("loss")}\nplan={trainer.plan.as_dict()}\nmax_steps={steps}\n'
            f'visible_devices={names}\n'
        )
    try:
        trainer.fit(train, valid)
        if os.environ.get('AUDIO_DB_TB_DDP_SAVE_PARAMETERS') == '1':
            from src.training.core.distributed import unwrap
            parameters = {key: parameter.detach().cpu() for key, parameter in unwrap(task.generator).named_parameters()}
            torch.save(parameters, root / f'parameters-{rank}.pt')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
