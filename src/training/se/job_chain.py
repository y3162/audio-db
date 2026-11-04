"""Fail-closed, one-process training segments for a pre-submitted job chain."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import yaml

from src.training.core.checkpoint import assert_output_allowed, load_checkpoint
from src.training.core.metrics_log import write_json
from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from src.training.se.run import execute

ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def source_identity() -> dict:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Include untracked Python source; no assumption that working tree was committed.
    files = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts' / 'training').glob('*.sh'))
    sha = hashlib.sha256()
    for path in files:
        sha.update(str(path.relative_to(ROOT)).encode() + b'\0')
        sha.update(bytes.fromhex(digest(path)))
    return {'commit': commit, 'source_sha256': sha.hexdigest()}


def environment() -> dict:
    packages = {}
    for name in ('torch', 'torchaudio', 'numpy', 'duckdb', 'mamba-ssm', 'causal-conv1d', 'librosa', 'nnAudio'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()
    properties = torch.cuda.get_device_properties(0)
    return {'python': platform.python_version(), 'packages': packages, 'torch_cuda': torch.version.cuda,
            'driver': sorted(set(driver)), 'gpu_name': properties.name, 'gpu_memory': properties.total_memory}


def initialize(directory: Path, config_path: Path) -> None:
    directory = directory.resolve()
    assert_output_allowed(directory)
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    raw['checkpoint_dir'] = str(directory / 'checkpoints')
    directory.mkdir(parents=True, exist_ok=False)
    config = directory / 'config.yaml'
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    normalized = load_training_config(config)
    database = Path(normalized['data']['database']).resolve()
    meta = {'format': 1, 'created': now(), 'config_sha256': digest(config),
            'database_sha256': digest(database),
            'model': normalized['model']['name'], 'seed': normalized['seed'], **source_identity()}
    write_json(directory / 'experiment.json', meta)
    print(json.dumps(meta, indent=2))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_successful_job(job_id: str, *, timeout: float = 120) -> None:
    """A hold releases on failure too; require the scheduler accounting result."""
    if not job_id.isdecimal():
        raise TrainingError('previous scheduler job ID is missing or invalid')
    until = time.monotonic() + timeout
    while True:
        result = subprocess.run(['qacct', '-j', job_id], text=True, capture_output=True)
        fields = {}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] in ('jobnumber', 'failed', 'exit_status'):
                    fields[parts[0]] = parts[1]
            if fields.get('jobnumber') == job_id and 'failed' in fields and 'exit_status' in fields:
                if fields['failed'] != '0' or fields['exit_status'] != '0':
                    raise TrainingError(f'previous scheduler job failed: {fields}')
                return
        if time.monotonic() >= until:
            raise TrainingError(f'previous job accounting unavailable: {job_id}')
        time.sleep(5)


def run(directory: Path, sequence: int, seconds: float, segment_steps: int | None = None) -> None:
    directory = directory.resolve()
    if sequence < 1 or seconds <= 0:
        raise TrainingError('sequence and seconds must be positive')
    with (directory / '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TrainingError('experiment is already running') from exc
        state_path = directory / f'job-{sequence:03d}.json'
        if state_path.exists():
            raise TrainingError(f'job sequence already recorded: {sequence}')
        started = time.time()
        state = {'status': 'RUNNING', 'sequence': sequence, 'job_id': os.getenv('JOB_ID'),
                 'host': platform.node(), 'start_time': now(), 'start_step': 0, 'start_epoch': 0}
        write_json(state_path, state)
        try:
            meta = json.loads((directory / 'experiment.json').read_text())
            config_path = directory / 'config.yaml'
            if digest(config_path) != meta['config_sha256'] or source_identity() != {k: meta[k] for k in ('commit', 'source_sha256')}:
                raise TrainingError('configuration, commit, or training source changed')
            config = load_training_config(config_path)
            if digest(Path(config['data']['database']).resolve()) != meta['database_sha256']:
                raise TrainingError('training database changed')
            env = environment()
            state['environment'] = env
            state['cuda_visible_devices'] = os.getenv('CUDA_VISIBLE_DEVICES')
            resume = None
            if sequence > 1:
                previous = json.loads((directory / f'job-{sequence-1:03d}.json').read_text())
                if previous.get('status') not in ('COMPLETED', 'CHECKPOINTED'):
                    raise TrainingError('previous job is not checkpointed successfully')
                previous_job = previous.get('job_id')
                if os.getenv('JOB_ID') and previous_job != os.getenv('JOB_ID'):
                    require_successful_job(str(previous_job or ''))
                if previous.get('status') == 'COMPLETED':
                    state.update(status='COMPLETED', reason='previous job completed', start_step=previous['end_step'],
                                 end_step=previous['end_step'], start_epoch=previous['end_epoch'], end_epoch=previous['end_epoch'])
                    return
                if previous['environment'] != env:
                    raise TrainingError('execution environment changed')
                resume = directory / previous['checkpoint']
                if digest(resume) != previous['checkpoint_sha256']:
                    raise TrainingError('checkpoint checksum mismatch')
                payload = load_checkpoint(resume, torch.device('cpu'))
                if payload['config'] != config or payload['global_step'] != previous['end_step'] or payload['epoch'] != previous['end_epoch']:
                    raise TrainingError('checkpoint configuration or progress mismatch')
                state.update(start_step=payload['global_step'], start_epoch=payload['epoch'])
            write_json(state_path, state)
            trainer = execute(config_path, resume=resume, deadline=started + seconds,
                              segment_steps=segment_steps, strict_resume=True)
            if trainer.stop_reason not in ('CHECKPOINTED', 'COMPLETED') or not trainer.last_checkpoint:
                raise TrainingError('trainer returned without a verified final checkpoint')
            checkpoint = Path(trainer.last_checkpoint)
            payload = load_checkpoint(checkpoint, torch.device('cpu'))
            if payload['global_step'] != trainer.global_step or trainer.global_step <= state['start_step']:
                raise TrainingError('checkpoint progress did not advance')
            state.update(status=trainer.stop_reason, reason=trainer.stop_reason.lower(),
                         end_step=trainer.global_step, end_epoch=payload['epoch'],
                         checkpoint=str(checkpoint.relative_to(directory)), checkpoint_sha256=digest(checkpoint))
        except BaseException as exc:
            state.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
            raise
        finally:
            state.update(end_time=now(), elapsed_seconds=round(time.time() - started, 3),
                         exit_status=0 if state['status'] in ('CHECKPOINTED', 'COMPLETED') else 1)
            write_json(state_path, state)


def submit(directory: Path, group: str, count: int, h_rt: str, seconds: float,
           segment_steps: int | None) -> None:
    directory = directory.resolve()
    if count < 1 or seconds <= 0:
        raise TrainingError('count and seconds must be positive')
    try:
        hours, minutes, seconds_field = (int(part) for part in h_rt.split(':'))
        wall_limit = hours * 3600 + minutes * 60 + seconds_field
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds_field < 60:
            raise ValueError
    except ValueError as exc:
        raise TrainingError('h_rt must be HH:MM:SS') from exc
    if seconds > wall_limit - 60:
        raise TrainingError('internal stop must leave at least 60 seconds before h_rt')
    if not (directory / 'experiment.json').is_file():
        raise TrainingError('initialize experiment before submission')
    # Exclusive creation makes a partial submission visible and prevents an
    # accidental second chain; it never schedules beyond this explicit count.
    record_path = directory / 'submissions.jsonl'
    previous = None
    with record_path.open('x', encoding='utf-8') as records:
        for sequence in range(1, count + 1):
            cmd = ['qsub', '-terse', '-g', group, '-p', '-5', '-l', 'gpu_h=1',
                   '-l', f'h_rt={h_rt}', '-cwd', '-N', f'audiodb-{sequence:03d}',
                   '-o', str(directory / f'job-{sequence:03d}.out'),
                   '-e', str(directory / f'job-{sequence:03d}.err')]
            if previous:
                cmd.extend(['-hold_jid', previous])
            cmd.extend([str(ROOT / 'scripts/training/job_segment.sh'), '--directory', str(directory),
                        '--sequence', str(sequence), '--seconds', str(seconds)])
            if segment_steps is not None:
                cmd.extend(['--segment-steps', str(segment_steps)])
            result = subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT)
            job_id = result.stdout.strip()
            records.write(json.dumps({'sequence': sequence, 'status': 'PENDING' if result.returncode == 0 else 'FAILED',
                                      'command': cmd, 'job_id': job_id,
                                      'returncode': result.returncode, 'stderr': result.stderr}) + '\n')
            records.flush()
            os.fsync(records.fileno())
            if result.returncode or not job_id.isdecimal():
                raise TrainingError(f'qsub failed for sequence {sequence}: {result.stderr or job_id}')
            print(f'{sequence}: {job_id}', flush=True)
            previous = job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--directory', type=Path, required=True)
    init.add_argument('--config', type=Path, required=True)
    job = sub.add_parser('run')
    job.add_argument('--directory', type=Path, required=True)
    job.add_argument('--sequence', type=int, required=True)
    job.add_argument('--seconds', type=float, required=True)
    job.add_argument('--segment-steps', type=int)
    chain = sub.add_parser('submit')
    chain.add_argument('--directory', type=Path, required=True)
    chain.add_argument('--group', required=True)
    chain.add_argument('--count', type=int, required=True)
    chain.add_argument('--h-rt', required=True)
    chain.add_argument('--seconds', type=float, required=True)
    chain.add_argument('--segment-steps', type=int)
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.directory, args.config)
    elif args.command == 'run':
        run(args.directory, args.sequence, args.seconds, args.segment_steps)
    else:
        submit(args.directory, args.group, args.count, args.h_rt, args.seconds, args.segment_steps)


if __name__ == '__main__':
    main()
