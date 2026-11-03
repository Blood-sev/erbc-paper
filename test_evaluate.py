import copy
import json
import math
import unittest
from pathlib import Path
from evaluate import score


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(Path(__file__).with_name('synthetic_scores.json').read_text())

    def test_fixed_denominator_and_valid_only(self):
        result = score(self.data)
        self.assertAlmostEqual(result['pMAE'], 40)
        self.assertAlmostEqual(result['pRMSE'], 100 * math.sqrt(1.04 / 3))
        self.assertAlmostEqual(result['valid_MAE'], 10)
        self.assertAlmostEqual(result['valid_RMSE'], math.sqrt(200))

    def test_missing_or_duplicate_identity_rejected(self):
        for records in [self.data['records'][:-1], self.data['records'] + self.data['records'][:1]]:
            with self.assertRaises(ValueError):
                score(dict(self.data, records=records))

    def test_execution_failure_cannot_be_scored(self):
        self.data['records'][2]['status'] = 'execution_error'
        with self.assertRaises(ValueError):
            score(self.data)

    def test_invalid_completed_output_is_penalized(self):
        self.data['records'][2]['status'] = 'invalid'
        self.assertAlmostEqual(score(self.data)['pMAE'], 40)

    def test_invalid_numeric_values_rejected(self):
        for value in [float('nan'), float('inf'), -0.1, 1.1, True, '0.5']:
            for field in ['prediction', 'target']:
                data = copy.deepcopy(self.data)
                data['records'][0][field] = value
                with self.assertRaises(ValueError):
                    score(data)

    def test_no_valid_cases_reports_null(self):
        for row in self.data['records']:
            row['status'] = 'blocked'
        result = score(self.data)
        self.assertEqual(result['pMAE'], 100)
        self.assertIsNone(result['valid_MAE'])

    def test_record_order_does_not_change_scores(self):
        self.assertEqual(score(self.data), score(dict(self.data, records=list(reversed(self.data['records'])))))


if __name__ == '__main__':
    unittest.main()
