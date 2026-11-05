"""Convert the primary JSONL training record into TensorBoard scalars.

The conversion never modifies the source and refuses to reuse an output directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def scalars(record):
    """Yield (tag, value, optimizer step) for available scalar records."""
    kind = record.get('type')
    if kind not in ('train', 'validation') or 'global_step' not in record:
        return
    step = int(record['global_step'])

    def emit(tag, value):
        if isinstance(value, dict):
            if value.get('status') != 'ok':
                return
            value = value.get('value')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            yield tag, float(value), step

    for name, value in (record.get('losses') or {}).items():
        yield from emit(f'{kind}/loss/{name}', value)
    if kind == 'validation':
        for name in ('pesq', 'stoi', 'si_sdr', 'si_snr'):
            yield from emit(f'validation/{name}', record.get(name))
    else:
        for name, value in (record.get('learning_rate') or {}).items():
            yield from emit(f'learning_rate/after_update/{name}', value)
        for name in ('epoch', 'global_batch_size', 'micro_batch_size', 'gradient_accumulation_steps'):
            yield from emit(f'progress/{name}', record.get(name))


def open_writer(directory: Path, resume_step: int | None = None):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is enabled but tensorboard is not installed") from exc
    return SummaryWriter(log_dir=str(directory), purge_step=None if resume_step is None else resume_step + 1)


def write_run_info(writer, config, world_size, step):
    # Allowlist prevents paths, credentials and arbitrary YAML values leaking.
    import platform
    import subprocess
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=2, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = 'unavailable'
    info = {key: config.get(key) for key in ('seed', 'loss', 'global_batch_size', 'micro_batch_size_per_gpu', 'deterministic')}
    info['model'] = config.get('model', {}).get('name')
    allowed = (str, int, float, bool, type(None))
    for section in ('optimizer', 'scheduler'):
        info[section] = {k: v for k, v in config.get(section, {}).items()
                         if isinstance(v, allowed) or (isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v))}
    info['train_set'] = config.get('data', {}).get('train_set')
    info['validation_set'] = config.get('data', {}).get('validation_set')
    info.update(world_size=world_size, git_commit=commit, python=platform.python_version())
    writer.add_text('run/settings', json.dumps(info, ensure_ascii=False, sort_keys=True), step)


def write_scalars(writer, record):
    for tag, value, step in scalars(record):
        writer.add_scalar(tag, value, step)


def rebuild(source: Path, destination: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError('TensorBoard and PyTorch are required for conversion') from exc
    # Exclusive creation prevents silently mixing or overwriting existing runs.
    destination.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir=str(destination))
    try:
        with source.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    for tag, value, step in scalars(json.loads(line)):
                        writer.add_scalar(tag, value, step)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    rebuild(args.source, args.destination)


if __name__ == '__main__':
    main()
