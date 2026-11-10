"""Create a fresh LibriSpeech/DEMAND recipe set for future SEMamba experiments.

The existing resource DB is read-only. Staging and the new DB must not exist.
"""
import json
from pathlib import Path
import sys
import duckdb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.training_sets import _speech, _noise, plan_pairs, assert_disjoint, build_case, write_staging, TRAIN_SCENES, VALID_SCENES
from src.data.staging import rebuild_database

stage = ROOT / 'data/staging/semamba-medium-20261101'
db = ROOT / 'data/db/semamba-medium-20261101.duckdb'
if stage.exists() or db.exists():
    raise SystemExit('refusing to overwrite staging or database')
con = duckdb.connect(str(ROOT / 'data/db/resource.duckdb'), read_only=True)
try:
    train = plan_pairs(_speech(con, 'train-clean-100'), _noise(con, TRAIN_SCENES), role='train', count=64)
    valid = plan_pairs(_speech(con, 'dev-clean'), _noise(con, VALID_SCENES), role='validation', count=8)
finally:
    con.close()
assert_disjoint(train, valid)
cases = [build_case(pair, corpora_root=ROOT / 'data/corpora', database_parent=db.parent) for pair in train + valid]
sets = [('semamba_medium_train', 'LibriSpeech train-clean-100 and DEMAND, 64 speakers, 2 second recipes.', [c['case_id'] for c in cases[:64]]), ('semamba_medium_valid', 'LibriSpeech dev-clean and disjoint DEMAND scenes, 8 speakers, 2 second recipes.', [c['case_id'] for c in cases[64:]])]
stage.mkdir(parents=True, exist_ok=False)
write_staging(stage, cases, sets)
report = {'train_cases': len(train), 'valid_cases': len(valid), 'train_speakers': sorted({x['speech']['speaker_id'] for x in train}), 'valid_speakers': sorted({x['speech']['speaker_id'] for x in valid}), 'train_noise': sorted({x['noise']['path'] for x in train}), 'valid_noise': sorted({x['noise']['path'] for x in valid}), 'snr_db': [0,5,10,15], 'protocol': 'libri-demand-smoke-v1; medium subset; not VoiceBank-DEMAND', 'held_out_test_used': False, 'database': str(db)}
(stage / 'leak_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
report['registered'] = rebuild_database(stage, db)
print(json.dumps(report, indent=2))
