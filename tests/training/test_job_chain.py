import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer, request_training_stop, reset_training_stop
from src.training.se import job_chain
from tests.training.mock_task import LinearPairTask
from tests.training.support import cpu_process, paired_dataset, tiny_config


class JobChainTests(unittest.TestCase):
    def test_update_boundary_resume_keeps_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = paired_dataset(root / 'wav', 8, frames=320, segment_size=80, seed=7)
            def trainer(name):
                config = tiny_config(root / name, seed=7, loss='mock', max_steps=4,
                                     global_batch_size=4, micro_batch_size_per_gpu=2)
                task = LinearPairTask(config, torch.device('cpu'))
                return task, Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=4,
                                     seed=7, checkpoint_dir=root / name, checkpoint_interval=0,
                                     validation_interval=0, config=config)
            full_task, full = trainer('full')
            full.fit(dataset)
            first_task, first = trainer('split')
            first.fit(dataset, segment_steps=2)
            self.assertEqual(first.stop_reason, 'CHECKPOINTED')
            second_task, second = trainer('split')
            second.fit(dataset, resume=Path(first.last_checkpoint), segment_steps=2)
            self.assertEqual(second.stop_reason, 'COMPLETED')
            self.assertEqual(first_task.case_ids + second_task.case_ids, full_task.case_ids)
            self.assertEqual(second.global_step, full.global_step)
            for key, value in full_task.generator_state_dict().items():
                self.assertTrue(torch.equal(value, second_task.generator_state_dict()[key]))

    def test_failed_predecessor_and_corrupt_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.yaml'
            config.write_text('placeholder')
            db = root / 'db'
            db.write_bytes(b'database')
            identity = {'commit': 'a', 'source_sha256': 'b'}
            meta = {'config_sha256': job_chain.digest(config), 'database_sha256': job_chain.digest(db), **identity}
            (root / 'experiment.json').write_text(json.dumps(meta))
            env = {'driver': 'same'}
            with patch.object(job_chain, 'source_identity', return_value=identity), \
                 patch.object(job_chain, 'environment', return_value=env), \
                 patch.object(job_chain, 'load_training_config', return_value={'data': {'database': str(db)}}), \
                 patch.object(job_chain, 'execute') as execute:
                (root / 'job-001.json').write_text(json.dumps({'status': 'FAILED'}))
                with self.assertRaisesRegex(Exception, 'previous job'):
                    job_chain.run(root, 2, 10)
                self.assertEqual(json.loads((root / 'job-002.json').read_text())['status'], 'FAILED')
                (root / 'job-002.json').write_text(json.dumps({'status': 'CHECKPOINTED', 'environment': env,
                                                               'job_id': os.getenv('JOB_ID'), 'checkpoint': 'bad.pt', 'checkpoint_sha256': 'bad'}))
                (root / 'bad.pt').write_bytes(b'corrupt')
                with self.assertRaisesRegex(Exception, 'checksum'):
                    job_chain.run(root, 3, 10)
                execute.assert_not_called()

    def test_scheduler_failure_is_rejected(self):
        from subprocess import CompletedProcess
        record = 'jobnumber 123\nfailed 0\nexit_status 1\n'
        with patch.object(job_chain.subprocess, 'run', return_value=CompletedProcess([], 0, record, '')):
            with self.assertRaisesRegex(Exception, 'scheduler job failed'):
                job_chain.require_successful_job('123')

    def test_submit_records_finite_chain_and_dependency(self):
        from subprocess import CompletedProcess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'experiment.json').write_text('{}')
            replies = [CompletedProcess([], 0, '11\n', ''), CompletedProcess([], 0, '12\n', '')]
            with patch.object(job_chain.subprocess, 'run', side_effect=replies) as qsub:
                job_chain.submit(root, 'group', 2, '00:10:00', 480, 2)
            first, second = (call.args[0] for call in qsub.call_args_list)
            self.assertNotIn('-hold_jid', first)
            self.assertEqual(second[second.index('-hold_jid') + 1], '11')
            records = [json.loads(line) for line in (root / 'submissions.jsonl').read_text().splitlines()]
            self.assertEqual([r['job_id'] for r in records], ['11', '12'])
            with self.assertRaises(FileExistsError):
                job_chain.submit(root, 'group', 2, '00:10:00', 480, 2)

    def test_lock_prevents_duplicate(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / '.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(Exception, 'already running'):
                    job_chain.run(root, 1, 10)

    def test_lock_file_does_not_block_a_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = paired_dataset(root / 'wav', 8, frames=320, segment_size=80, seed=7)
            config = tiny_config(root / 'ckpt', seed=7, loss='mock', max_steps=1,
                                 global_batch_size=4, micro_batch_size_per_gpu=2)
            (root / 'ckpt').mkdir()
            (root / 'ckpt' / '.lock').write_text('held by this test')
            task = LinearPairTask(config, torch.device('cpu'))
            trainer = Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=1,
                              seed=7, checkpoint_dir=root / 'ckpt', checkpoint_interval=0,
                              validation_interval=0, config=config)
            trainer.fit(dataset)
            self.assertEqual(trainer.global_step, 1)

    def test_deadline_stops_after_first_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = paired_dataset(root / 'wav', 8, frames=320, segment_size=80, seed=7)
            config = tiny_config(root / 'ckpt', seed=7, loss='mock', max_steps=4,
                                 global_batch_size=4, micro_batch_size_per_gpu=2)
            task = LinearPairTask(config, torch.device('cpu'))
            trainer = Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=4,
                              seed=7, checkpoint_dir=root / 'ckpt', checkpoint_interval=0,
                              validation_interval=0, config=config)
            trainer.fit(dataset, deadline=time.time() - 1)
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(trainer.stop_reason, 'CHECKPOINTED')
            self.assertTrue(Path(trainer.last_checkpoint).is_file())

    def test_sigterm_stops_after_current_update_and_completed_resume_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = paired_dataset(root / 'wav', 8, frames=320, segment_size=80, seed=7)
            config = tiny_config(root / 'ckpt', seed=7, loss='mock', max_steps=4,
                                 global_batch_size=4, micro_batch_size_per_gpu=2)
            task = LinearPairTask(config, torch.device('cpu'))
            trainer = Trainer(task, plan_batches(4, 2, 1), cpu_process(), max_steps=4,
                              seed=7, checkpoint_dir=root / 'ckpt', checkpoint_interval=0,
                              validation_interval=0, config=config)
            original = task.train_update
            def stop_after_update(*args, **kwargs):
                result = original(*args, **kwargs)
                request_training_stop(signal.SIGTERM, None)
                return result
            task.train_update = stop_after_update
            try:
                trainer.fit(dataset)
            finally:
                reset_training_stop()
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(trainer.stop_reason, 'CHECKPOINTED')
            self.assertFalse((root / 'ckpt' / 'complete.json').is_file())
            finished = Trainer(LinearPairTask(config, torch.device('cpu')), plan_batches(4, 2, 1),
                               cpu_process(), max_steps=4, seed=7, checkpoint_dir=root / 'done',
                               checkpoint_interval=0, validation_interval=0, config=tiny_config(
                                   root / 'done', seed=7, loss='mock', max_steps=4,
                                   global_batch_size=4, micro_batch_size_per_gpu=2))
            finished.fit(dataset)
            self.assertEqual(finished.stop_reason, 'COMPLETED')
            self.assertTrue((root / 'done' / 'complete.json').is_file())
            again = Trainer(LinearPairTask(config, torch.device('cpu')), plan_batches(4, 2, 1),
                            cpu_process(), max_steps=4, seed=7, checkpoint_dir=root / 'done',
                            checkpoint_interval=0, validation_interval=0, config=finished.config)
            with self.assertRaisesRegex(Exception, 'already complete'):
                again.fit(dataset, resume=Path(finished.last_checkpoint))

    def test_second_process_cannot_lock_the_same_experiment(self):
        from src.training.se.run import experiment_lock
        with tempfile.TemporaryDirectory() as tmp:
            held = experiment_lock(Path(tmp))
            try:
                probe = subprocess.run(
                    [sys.executable, "-c",
                     "from pathlib import Path; from src.training.se.run import experiment_lock; "
                     f"experiment_lock(Path({tmp!r}))"],
                    cwd=os.getcwd(), capture_output=True, text=True, check=False,
                )
            finally:
                held.close()
            self.assertNotEqual(probe.returncode, 0)
            self.assertIn("already running", probe.stderr)


if __name__ == '__main__':
    unittest.main()
