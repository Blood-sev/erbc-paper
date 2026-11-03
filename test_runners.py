import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from erbc_io import read
from run_api import classify, messages, run
from run_numerical import ensemble
from score_predictions import score

ROOT = Path(__file__).resolve().parent


class RunnerContracts(unittest.TestCase):
    def setUp(self):
        self.config = read(ROOT / 'api_config.json')
        self.data = read(ROOT / 'example_api_pipeline.json')

    def test_dry_run_never_calls_service(self):
        report = run(self.data, self.config, None, 'pipeline')
        self.assertEqual(report['network_requests'], 0)

    def test_own_b_binding_and_resume(self):
        calls = []
        def service(payload, config):
            calls.append(copy.deepcopy(payload))
            value = '0.2' if len(calls) <= 2 else '0.8'
            return {'model': payload['model'], 'choices': [
                {'finish_reason': 'stop', 'message': {'content': value}}]}
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'run'
            result = run(self.data, self.config, out, 'pipeline', True, service)
            self.assertEqual(len(calls), 4)
            import json
            self.assertEqual(json.loads(calls[1]['messages'][1]['content'])['upstream_B_prediction']['value'], .2)
            self.assertEqual(json.loads(calls[3]['messages'][1]['content'])['upstream_B_prediction']['value'], .8)
            self.assertEqual(run(self.data, self.config, out, 'pipeline', True, service), result)
            self.assertEqual(len(calls), 4)

    def test_invalid_b_prevents_c_call(self):
        calls = []
        def service(payload, config):
            calls.append(payload)
            return {'model': payload['model'], 'choices': [
                {'finish_reason': 'stop', 'message': {'content': 'not a score'}}]}
        with tempfile.TemporaryDirectory() as t:
            result = run(self.data, self.config, Path(t) / 'run', 'pipeline', True, service)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(x['records'][0]['status'] == 'invalid' for x in result.values()))

    def test_transport_or_budget_failure_not_model_penalty(self):
        for raw in [{}, {'model': 'm', 'choices': [
                {'finish_reason': 'length', 'message': {'content': '.5'}}]}]:
            with self.assertRaises(RuntimeError):
                classify(raw, 'm')

    def test_evidence_is_preserved(self):
        import json
        row = self.data['records'][0]
        new = messages(row, 'C', 'pipeline', self.config, {'id': row['id'], 'status': 'valid', 'prediction': .25})
        old = json.loads(row['C_messages'][1]['content'])
        body = json.loads(new[1]['content'])
        self.assertTrue(all(body[k] == v for k, v in old.items()))

    def test_label_fields_rejected(self):
        self.data['records'][0]['target'] = .2
        with self.assertRaises(ValueError):
            run(self.data, self.config, None, 'pipeline')

    def test_ensemble_requires_same_tokens(self):
        a = {'expected_ids': ['demo'], 'records': [
            {'id': 'demo', 'status': 'valid', 'prediction': .2, 'token_sha256': 'a'}]}
        b = copy.deepcopy(a)
        b['records'][0]['prediction'] = .8
        self.assertAlmostEqual(ensemble(a, b)['records'][0]['prediction'], .5)
        b['records'][0]['token_sha256'] = 'b'
        with self.assertRaises(ValueError):
            ensemble(a, b)

    def test_join_requires_exact_label_membership(self):
        p = {'expected_ids': ['a'], 'records': [{'id': 'a', 'status': 'valid', 'prediction': .5}]}
        self.assertEqual(score(p, {'records': [{'id': 'a', 'target': .5}]})['pMAE'], 0)
        with self.assertRaises(ValueError):
            score(p, {'records': [{'id': 'b', 'target': .5}]})


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),
                     'Install requirements.txt for head tests')
class HeadContracts(unittest.TestCase):
    def test_zero_head_preserves_reference(self):
        from predict_erbc import predict
        data = read(ROOT / 'example_inference.json')
        result = predict(data)
        self.assertEqual(result['records'][0]['prediction'], data['records'][0]['state']['prediction'])
        self.assertEqual(result['records'][0]['delta_q'], 0)
        self.assertEqual(result['records'][0]['delta_b'], 0)

    def test_invalid_mass_and_hidden_labels_rejected(self):
        from predict_erbc import predict
        data = read(ROOT / 'example_inference.json')
        data['records'][0]['state']['mass'] = [.4, .7]
        with self.assertRaises(ValueError):
            predict(data)
        data = read(ROOT / 'example_inference.json')
        data['records'][0]['target'] = .5
        with self.assertRaises(ValueError):
            predict(data)

    def test_three_stage_training_and_condition_binding(self):
        from train_erbc import train
        from predict_erbc import predict
        with tempfile.TemporaryDirectory() as t:
            runs = train(ROOT / 'example_training.json', ROOT / 'example_training_config.json', Path(t) / 'fit')
            self.assertEqual(len(runs), 6)
            self.assertEqual(sum(x['steps'] for x in runs), 144)
            data = read(ROOT / 'example_inference.json')
            for r in runs:
                checkpoint = Path(t) / 'fit' / r['checkpoint']
                result = predict(data, checkpoint, r['arm'])
                self.assertTrue(0 <= result['records'][0]['prediction'] <= 1)
                if r['arm'] == 'q_only':
                    self.assertEqual(result['records'][0]['delta_b'], 0)
                if r['arm'] == 'bias_only':
                    self.assertEqual(result['records'][0]['delta_q'], 0)
            with self.assertRaises(ValueError):
                predict(data, Path(t) / 'fit' / runs[0]['checkpoint'], 'joint')


if __name__ == '__main__':
    unittest.main()
