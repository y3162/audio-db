"""Stage the complete official paired test split in a separate database."""
import json
import sys
from pathlib import Path

import torchaudio

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.recipe import canonical_recipe, case_content_hash, case_id_for, sha256_file
from src.data.staging import rebuild_database
from src.data.training_sets import write_staging


def main() -> None:
    corpus = ROOT / 'data/corpora/VoiceBank-DEMAND'
    stage = ROOT / 'data/staging/semamba-voicebank-test-20261101-v2'
    db = ROOT / 'data/db/semamba-voicebank-test-20261101-v2.duckdb'
    if stage.exists() or db.exists():
        raise SystemExit('refusing to overwrite staging or database')
    clean = corpus / 'clean_testset_wav'
    noisy = corpus / 'noisy_testset_wav'
    names = sorted(p.name for p in clean.glob('*.wav'))
    if not names or set(names) != {p.name for p in noisy.glob('*.wav')}:
        raise SystemExit('missing or unmatched clean/noisy files')
    cases = []
    for name in names:
        a, b = clean / name, noisy / name
        info_a, info_b = torchaudio.info(a), torchaudio.info(b)
        if (info_a.num_frames, info_a.sample_rate, info_a.num_channels) != (info_b.num_frames, info_b.sample_rate, info_b.num_channels):
            raise SystemExit(f'paired audio shape mismatch: {name}')
        sources = {kind: {'path': f'../corpora/VoiceBank-DEMAND/{p.parent.name}/{name}', 'sha256': sha256_file(p)} for kind, p in [('clean', a), ('noisy', b)]}
        recipe = canonical_recipe({'recipe_version': 1, 'sources': sources, 'operations': [], 'output': {'sample_rate': 16000, 'reference': 'dry_clean'}})
        meta = {'corpus': 'VoiceBank-DEMAND', 'original_filename': name, 'pairing': 'official matching filenames', 'checkpoint_training_overlap': 'unknown'}
        body = {'kind': 'paired', 'dataset': 'VoiceBank-DEMAND', 'split': 'test', 'sample_rate': 16000, 'transcript': None, 'speaker_id': name.split('_')[0], 'recipe': recipe, 'metadata': meta}
        cases.append({'case_id': case_id_for(recipe), 'recipe_json': json.dumps(recipe, sort_keys=True), 'content_hash': case_content_hash(body), 'metadata': json.dumps(meta, sort_keys=True), **{k: body[k] for k in ('kind', 'dataset', 'split', 'sample_rate', 'transcript', 'speaker_id')}})
    write_staging(stage, cases, [('voicebank_demand_official_test', 'Full official VoiceBank-DEMAND test split; paired by filename.', [c['case_id'] for c in cases])])
    report = rebuild_database(stage, db)
    print(json.dumps({'cases': len(cases), 'speakers': sorted({c['speaker_id'] for c in cases}), 'registered': report}, indent=2))


if __name__ == '__main__':
    main()
