import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'tests/training/four_process/compare.py'
RESUME=ROOT/'tests/training/four_process/mock_resume.py'
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def command(*args):
    return subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=ENV,text=True,capture_output=True,timeout=180)

class FourProcessTest(unittest.TestCase):
    def test_success_order_report_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a'; b=Path(tmp)/'b'
            for path, order in [(a,[]),(b,['--order','independent_b','official_b','independent_a','official_a'])]:
                p=command(SCRIPT,'run',path,'--steps','2',*order); self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            x=json.loads((a/'report.json').read_text()); y=json.loads((b/'report.json').read_text())
            self.assertEqual(len(set(x['pids'].values())),4); self.assertTrue(x['valid']); self.assertEqual(x['comparisons'],y['comparisons'])
            p=command(SCRIPT,'report',a); self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            import torch
            result_path=a/'official_a/result.pt'; corrupted=torch.load(result_path,weights_only=True)
            corrupted['records'][0]['loss'] += 1; torch.save(corrupted,result_path)
            p=command(SCRIPT,'report',a); self.assertNotEqual(p.returncode,0)
    def test_faults(self):
        for fault in ('input','weight','optimizer','gradient','exception','missing','nan','rng','case'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                p=command(SCRIPT,'run',Path(tmp)/'run','--fault',fault)
                self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
                report=json.loads((Path(tmp)/'run/report.json').read_text())
                self.assertFalse(report['valid'])
                expected={'input':'conditions mismatch','weight':'conditions mismatch','optimizer':'conditions mismatch','gradient':'mock numerical mismatch','exception':'exit','missing':'FileNotFoundError','nan':'exit','rng':'conditions mismatch','case':'conditions mismatch'}[fault]
                self.assertTrue(any(expected in error for error in report['errors']),report['errors'])
                if fault == 'nan': self.assertIn('nonfinite loss',(Path(tmp)/'run/independent_b/stderr.log').read_text())
    def test_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'continuous.pt'; b=Path(tmp)/'resume.pt'
            p=command(RESUME,a,'4','new'); self.assertEqual(p.returncode,0,p.stderr)
            p=command(RESUME,b,'2','new'); self.assertEqual(p.returncode,0,p.stderr)
            p=command(RESUME,b,'4','resume'); self.assertEqual(p.returncode,0,p.stderr)
            from tests.training.four_process.compare import digest
            import torch
            aa=torch.load(a,weights_only=True); bb=torch.load(b,weights_only=True)
            for key in ('model','optimizer','scheduler','rng','data_position'): self.assertEqual(digest(aa[key]),digest(bb[key]),key)
    def test_worker_checkpoint_resume_matches_continuous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            continuous=root/'continuous'; split=root/'split'
            p=command(SCRIPT,'run',continuous,'--steps','4'); self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            p=command(SCRIPT,'run',split,'--steps','2'); self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            p=command(SCRIPT,'resume',split,'--steps','4'); self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            import torch
            from tests.training.four_process.compare import digest
            for name in ('official_a','official_b','independent_a','independent_b'):
                full=torch.load(continuous/name/'result.pt',weights_only=False)
                resumed=torch.load(split/name/'result.pt',weights_only=False)
                self.assertEqual([row['loss'] for row in full['records']],[row['loss'] for row in resumed['records']])
                self.assertEqual(full['global_step'], resumed['global_step'])
                self.assertEqual(full['data_position'], resumed['data_position'])
                for key in ('modules','optimizer','scheduler','rng_full_hash','rng_parts'):
                    self.assertEqual(digest(full[key]), digest(resumed[key]), key)
                self.assertEqual(full['rng_parts']['python'], resumed['rng_parts']['python'])
                self.assertEqual(full['rng_parts']['numpy'], resumed['rng_parts']['numpy'])
                self.assertEqual(full['rng_parts']['torch'], resumed['rng_parts']['torch'])
    def test_resume_rejects_changed_seed_and_detects_rng_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            p=command(SCRIPT,'run',root,'--steps','2'); self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            cfg=json.loads((root/'config.json').read_text()); cfg['seed']=cfg['seed']+1
            (root/'config.json').write_text(json.dumps(cfg))
            p=command(SCRIPT,'resume',root,'--steps','4')
            self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertTrue((root/'official_a'/'stderr.log').read_text().find('refusing to resume')>=0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trusted=root/'trusted'; tampered=root/'tampered'
            p=command(SCRIPT,'run',trusted,'--steps','4'); self.assertEqual(p.returncode,0,p.stderr)
            p=command(SCRIPT,'run',tampered,'--steps','2'); self.assertEqual(p.returncode,0,p.stderr)
            import torch
            checkpoint=tampered/'official_a'/'checkpoint.pt'
            state=torch.load(checkpoint,weights_only=False)
            import random as random_module
            import numpy as np
            held_torch, held_py, held_np = torch.get_rng_state(), random_module.getstate(), np.random.get_state()
            torch.manual_seed(99); random_module.seed(99); np.random.seed(99)
            state['rng']['torch']=torch.get_rng_state()
            state['rng']['python']=random_module.getstate()
            state['rng']['numpy']=np.random.get_state()
            torch.set_rng_state(held_torch); random_module.setstate(held_py); np.random.set_state(held_np)
            torch.save(state, checkpoint)
            p=command(SCRIPT,'resume',tampered,'--steps','4')
            self.assertNotEqual(p.returncode,0,p.stderr+p.stdout)
            report=json.loads((tampered/'report.json').read_text())
            self.assertTrue(any('mock numerical mismatch' in error for error in report['errors']), report['errors'])
            full=torch.load(trusted/'official_a'/'result.pt',weights_only=False)
            bad=torch.load(tampered/'official_a'/'result.pt',weights_only=False)
            self.assertNotEqual(full['rng_parts']['python'], bad['rng_parts']['python'])
            self.assertNotEqual(full['rng_parts']['numpy'], bad['rng_parts']['numpy'])
            self.assertNotEqual(full['rng_parts']['torch'], bad['rng_parts']['torch'])
    def test_voicebank_recipe_inputs_match_across_processes(self):
        db=ROOT/'data/db/voicebank.duckdb'
        self.assertTrue(db.is_file(), db)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'voicebank'
            p=command(SCRIPT,'run',path,'--model','mock','--steps','1','--dataset','voicebank_28_train','--database',db,'--segment-size','8','--sample-rate','16000')
            self.assertEqual(p.returncode,0,p.stderr+p.stdout)
            report=json.loads((path/'report.json').read_text())
            self.assertTrue(report['valid'], report['errors'])
            batch=report['comparisons']['official_a vs independent_a'][0]['batch'][0]
            self.assertEqual(len(batch['case_id']),64)
            self.assertEqual(batch['case_id'], batch['recipe_sha256'])
            self.assertEqual(batch['sample_rate'],16000)
            self.assertGreater(batch['clean_length'],8)
            self.assertEqual(batch['clean_length'], batch['noisy_length'])
            self.assertIsInstance(batch['crop_start'], int)
            self.assertTrue(batch['sources'])
            self.assertEqual(len(batch['clean_sha256']),64)
            self.assertEqual(len(batch['noisy_sha256']),64)
            self.assertTrue(report['comparisons']['official_a vs official_b'][0]['rng_parts_match'])

    def test_recipe_resume_prepares_distinct_accumulation_batches(self):
        from tests.training.four_process.recipes import planned_indices
        cfg = {'steps': 2, 'prepare_steps': 4, 'model': 'mossformergan', 'seed': 3, 'epoch': 0}
        indices = planned_indices(range(20), cfg)
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(set(indices)), 8)
        cfg['steps'] = 4
        self.assertEqual(indices, planned_indices(range(20), cfg))

    def test_fork_checkpoint_resumes_from_continuous_state(self):
        import shutil
        import torch
        from tests.training.four_process.compare import digest
        with tempfile.TemporaryDirectory() as tmp:
            continuous = Path(tmp) / 'continuous' / 'run'
            resumed = Path(tmp) / 'resumed' / 'run'
            p = command(SCRIPT, 'run', continuous, '--steps', '4', '--fork-step', '2')
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
            (resumed / 'official_a').mkdir(parents=True)
            for name in ('config.json', 'input.pt'):
                shutil.copyfile(continuous / name, resumed / name)
            shutil.copyfile(continuous / 'official_a/fork-step-2.pt', resumed / 'official_a/checkpoint.pt')
            p = command(SCRIPT, 'worker', resumed, 'official_a')
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
            a = torch.load(continuous / 'official_a/result.pt', weights_only=False)
            b = torch.load(resumed / 'official_a/result.pt', weights_only=False)
            for key in ('modules', 'optimizer', 'scheduler', 'rng_full_hash'):
                self.assertEqual(digest(a[key]), digest(b[key]), key)

    def test_deterministic_probe_setting_is_part_of_checkpoint_identity(self):
        from tests.training.four_process.compare import identity
        self.assertNotEqual(identity({'model': 'deepfilternet3'}),
                            identity({'model': 'deepfilternet3', 'independent_deterministic': True}))
        p = command(SCRIPT, 'run', '/unused', '--model', 'mock', '--independent-deterministic')
        self.assertNotEqual(p.returncode, 0)
    def test_mp_senet_update_order_and_two_optimizers(self):
        from tests.training.four_process.adapters import ADAPTERS
        adapter=ADAPTERS['mp_senet']
        self.assertEqual(adapter.update_order, ('discriminator','generator'))
        modules=adapter.build('independent', __import__('torch').device('cpu'))
        optimizers=adapter.optimizers(modules, 0.001)
        schedulers=adapter.schedulers(optimizers)
        self.assertEqual(set(optimizers), {'generator','discriminator'})
        self.assertEqual(set(schedulers), {'generator','discriminator'})
    def test_registered_update_orders(self):
        from tests.training.four_process.adapters import ADAPTERS
        self.assertEqual(ADAPTERS['semambapp'].update_order, ('discriminator','generator'))
        self.assertEqual(ADAPTERS['mossformergan'].update_order, ('generator','discriminator'))
        self.assertEqual(ADAPTERS['deepfilternet3'].update_order, ('generator',))
        self.assertEqual(ADAPTERS['semambapp'].sample_rate, 16000)
        self.assertEqual(ADAPTERS['mossformergan'].sample_rate, 16000)
        self.assertEqual(ADAPTERS['deepfilternet3'].sample_rate, 48000)
        self.assertIsNone(ADAPTERS['semambapp'].clip)
        self.assertEqual(ADAPTERS['mossformergan'].clip, 10.0)
        self.assertEqual(ADAPTERS['deepfilternet3'].clip, 1.0)
if __name__=='__main__': unittest.main()
