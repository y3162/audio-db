"""Paired comparisons never mix different valid utterance sets."""
import unittest

from src.evaluation.results.summary import build_summary


def record(case_id, noisy, enhanced):
    return {
        'status': 'ok', 'utterance_id': case_id,
        'identity': {'config_hash': 'hash'},
        'model': {'name': 'model', 'variant': None}, 'snr_db': None,
        'pesq': {'values': {'noisy': noisy, 'enhanced': enhanced}},
    }


def score(value):
    return {'status': 'ok', 'value': value}


class PairedSummaryTests(unittest.TestCase):
    def test_missing_enhanced_excludes_same_noisy_case(self):
        records = [record('a', score(1.0), score(2.0)),
                   record('b', score(4.0), {'status': 'na', 'reason': 'split', 'value': None}),
                   record('c', score(3.0), score(4.0))]
        summary = build_summary(records)
        groups = {g['condition']: g for g in summary['groups']}
        self.assertEqual(groups['noisy']['pesq_count'], 3)
        self.assertEqual(groups['enhanced']['pesq_count'], 2)
        paired = summary['paired_signal_metrics'][0]
        self.assertEqual(paired['count'], 2)
        self.assertEqual(paired['case_ids'], ['a', 'c'])
        self.assertEqual(paired['noisy_mean'], 2.0)
        self.assertEqual(paired['enhanced_mean'], 3.0)
        self.assertEqual(paired['mean_difference'], 1.0)
        self.assertEqual(paired['excluded'], [{'utterance_id': 'b', 'noisy_reason': None, 'enhanced_reason': 'split'}])

    def test_no_valid_pairs_never_fills_zero(self):
        summary = build_summary([record('a', score(float('nan')), score(1.0)),
                                 record('b', score(2.0), {'status': 'unavailable', 'value': None})])
        pair = summary['paired_signal_metrics'][0]
        self.assertEqual(pair['count'], 0)
        self.assertIsNone(pair['noisy_mean'])
        self.assertIsNone(pair['enhanced_mean'])
        self.assertIsNone(pair['mean_difference'])
        self.assertEqual([row['utterance_id'] for row in pair['excluded']], ['a', 'b'])
        self.assertEqual(pair['excluded'][0]['noisy_reason'], 'nonfinite_or_invalid')

    def test_missing_both_conditions_and_different_models(self):
        first = record('a', score(1.0), score(2.0))
        missing = record('b', score(3.0), score(4.0))
        missing['pesq']['values'] = {}
        other = record('c', score(10.0), score(20.0))
        other['model']['name'] = 'other'
        pairs = build_summary([first, missing, other])['paired_signal_metrics']
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]['count'], 1)
        self.assertEqual(pairs[0]['excluded'][0]['utterance_id'], 'b')
        self.assertEqual(pairs[0]['excluded'][0]['noisy_reason'], 'missing')
        self.assertEqual(pairs[1]['noisy_mean'], 10.0)


if __name__ == '__main__':
    unittest.main()
