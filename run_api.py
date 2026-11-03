"""Fixed scalar API baselines: C component or each model's own B-to-C chain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import urllib.error
from erbc_io import fraction, need, read, write

C_OLD = '\u8f93\u51fa\u5355\u4e2a\u6570\u5b57\u69fd0\u81f39\uff1a0\u8868\u793a0%\uff0c9\u8868\u793a100%\uff0c\u63090\u81f39\u7684\u6982\u7387\u5206\u5e03\u8ba1\u7b97\u8fde\u7eed\u671f\u671b\u3002'
C_NEW = '\u53ea\u8f93\u51fa\u4e00\u4e2a0\u52301\u4e4b\u95f4\u7684\u5341\u8fdb\u5236\u6570\u4f5c\u4e3a\u6700\u7ec8\u4f30\u8ba1\u503c\uff08\u4f8b\u59820.375\uff09\uff1b0\u8868\u793a0%\uff0c1\u8868\u793a100%\u3002\u4e0d\u8981\u8f93\u51fa\u767e\u5206\u53f7\u3001\u89e3\u91ca\u6216\u5176\u4ed6\u6587\u5b57\u3002'
B_NEW = '\u53ea\u8f93\u51fa\u4e00\u4e2a0\u52301\u4e4b\u95f4\u7684\u5341\u8fdb\u5236\u6570\u4f5c\u4e3a\u6574\u4f53\u64cd\u4f5c\u51cf\u8d1f\u7387\u7684\u6700\u7ec8\u4f30\u8ba1\u503c\uff08\u4f8b\u59820.375\uff09\uff1b0\u8868\u793a0%\uff0c1\u8868\u793a100%\u3002\u4e0d\u8981\u8f93\u51fa\u767e\u5206\u53f7\u3001\u89e3\u91ca\u6216\u5176\u4ed6\u6587\u5b57\u3002'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_score(text):
    if not isinstance(text, str) or not re.fullmatch(r'\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*', text):
        raise ValueError('Invalid scalar output')
    return fraction(float(text))


def messages(row, stage, mode, config, b=None):
    original = row[stage + '_messages']
    need(len(original) == 2 and [x['role'] for x in original] == ['system', 'user'],
         'Expected system/user message pair')
    result = [dict(m) for m in original]
    if stage == 'B':
        old, quiet = config['B_numeric_instruction'], config['B_no_text_instruction']
        need(result[0]['content'].count(old) == result[0]['content'].count(quiet) == 1,
             'B output instruction mismatch')
        result[0]['content'] = result[0]['content'].replace(quiet, '').replace(old, '\n' + B_NEW)
    else:
        need(result[0]['content'].count(C_OLD) == 1, 'C output instruction mismatch')
        result[0]['content'] = result[0]['content'].replace(C_OLD, C_NEW)
        if mode == 'pipeline':
            need(b is not None and b['status'] == 'valid' and b['id'] == row['id'],
                 'C requires the same unit\'s valid B result')
            value = fraction(b['prediction'])
            body = json.loads(original[1]['content'])
            need(set(body) == {'task', 'subject', 'question_segments', 'confirmed_blueprint',
                              'confirmed_operations', 'evidence', 'evidence_policy'},
                 'C evidence schema mismatch')
            body['upstream_B_prediction'] = {'quantity': 'holistic_operation_relief_fraction',
                                            'value': value, 'range': [0, 1], 'is_prediction': True}
            result[0]['content'] += config['C_link_instruction']
            result[1]['content'] = canonical(body).decode('utf-8')
    return result


def classify(raw, model):
    if not isinstance(raw, dict) or raw.get('model') != model or 'error' in raw or not raw.get('choices'):
        raise RuntimeError('API response identity/schema mismatch')
    choice = raw['choices'][0]
    if choice.get('finish_reason') not in ('stop', 'content_filter'):
        raise RuntimeError('API execution incomplete: output budget or finish reason')
    try:
        return {'status': 'valid', 'prediction': parse_score(choice.get('message', {}).get('content'))}
    except ValueError:
        return {'status': 'invalid'}


def send(payload, config):
    key = os.environ.get('ERBC_API_KEY')
    need(bool(key), 'Set ERBC_API_KEY in the terminal environment')
    endpoint = config['base_url'].rstrip('/') + '/chat/completions'
    need(endpoint.startswith('https://'), 'Use an HTTPS endpoint')
    for attempt in range(3):
        request = urllib.request.Request(endpoint, data=canonical(payload), headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=config['timeout']) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f'API execution incomplete: HTTP {error.code}') from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError('API execution incomplete: transport failure') from None
        time.sleep((2, 5)[attempt])


def run(data, config, output, mode, execute=False, transport=None):
    need(mode in ('component', 'pipeline'), 'Unknown protocol')
    need(set(data) == {'expected_ids', 'records'}, 'Unexpected input bundle fields')
    ids = data['expected_ids']
    need(ids and len(set(ids)) == len(ids) and all(isinstance(u, str) and u for u in ids),
         'Invalid expected membership')
    need(len(data['records']) == len(ids) and {r['id'] for r in data['records']} == set(ids),
         'Incomplete or duplicate input records')
    models = config['models']
    need(models and len(set(models)) == len(models), 'Invalid model list')
    need(config['temperature'] == 0 and config['max_tokens'] == 4096, 'Changed sampling settings')
    for r in data['records']:
        allowed = {'id', 'status', 'C_messages'} | ({'B_messages'} if mode == 'pipeline' else set())
        need(set(r) == allowed and r['status'] in ('valid', 'blocked'), 'Input row fields/status')
        if r['status'] == 'valid':
            # Validate every input before creating output or sending requests.
            messages(r, 'C', 'component', config)
            if mode == 'pipeline':
                messages(r, 'B', mode, config)
                messages(r, 'C', mode, config, {'id': r['id'], 'status': 'valid', 'prediction': .5})
    if not execute:
        return {'units': len(ids), 'ready': sum(r['status'] == 'valid' for r in data['records']),
                'models': models, 'protocol': mode, 'network_requests': 0}
    output = Path(output)
    binding = {'input_sha256': digest(data), 'config_sha256': digest(config), 'protocol': mode}
    if output.exists():
        need(read(output / 'binding.json') == binding, 'Output belongs to different inputs/settings')
    else:
        output.mkdir(parents=True)
        write(output / 'binding.json', binding)
    transport = transport or send
    all_results = {}
    for model in models:
        records = []
        for row in data['records']:
            if row['status'] == 'blocked':
                records.append({'id': row['id'], 'status': 'blocked'})
                continue
            b = None
            for stage in (('B', 'C') if mode == 'pipeline' else ('C',)):
                path = output / (digest([model, stage, row['id']]) + '.json')
                if stage == 'C' and b is not None and b['status'] == 'invalid':
                    records.append({'id': row['id'], 'status': 'invalid'})
                    break
                payload = {'model': model, 'messages': messages(row, stage, mode, config, b),
                           'temperature': 0, 'max_tokens': 4096, 'stream': False}
                request_binding = {'request_sha256': digest(payload), 'id': row['id'],
                                   'model': model, 'stage': stage, 'B_sha256': digest(b) if b else None}
                if path.exists():
                    saved = read(path)
                    need(saved['binding'] == request_binding, 'Cached request binding differs')
                    raw = saved['response']
                else:
                    raw = transport(payload, config)
                    classify(raw, model)
                    write(path, {'binding': request_binding, 'response': raw})
                result = {'id': row['id'], **classify(raw, model)}
                if stage == 'B':
                    b = result
                else:
                    records.append(result)
        all_results[model] = {'expected_ids': ids, 'records': records}
    dest = output / 'predictions.json'
    if dest.exists():
        need(read(dest) == all_results, 'Completed predictions differ')
    else:
        write(dest, all_results)
    return all_results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--config', default='api_config.json')
    p.add_argument('--mode', choices=['component', 'pipeline'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true', help='Send API requests; default is input validation')
    a = p.parse_args()
    result = run(read(a.input), read(a.config), a.output, a.mode, a.execute)
    if not a.execute:
        print(json.dumps(result, indent=2))
