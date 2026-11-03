"""Validated JSON interfaces for frozen evidence states."""
import json
import math
from pathlib import Path

ARMS = ('q_only', 'bias_only', 'joint')
STATE_KEYS = {'latent', 'responsibility', 'mass', 'prediction', 'gate'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def fraction(value):
    need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
         'Expected a finite fraction in [0, 1]')
    return value


def state(value):
    import torch
    need(set(value) == STATE_KEYS, 'Frozen-state fields differ from the input schema')
    # The cached latent is float32; the retained numerical state is float64.
    result = {k: torch.tensor(v, dtype=torch.float32 if k == 'latent' else torch.float64)
              for k, v in value.items()}
    n = result['mass'].numel()
    need(n > 0 and result['mass'].shape == (n,), 'Expected a nonempty mass vector')
    need(result['latent'].shape == (n, 32), 'Expected latent shape [operations, 32]')
    need(result['responsibility'].shape == (n, 3), 'Expected responsibility shape [operations, 3]')
    need(result['prediction'].ndim == result['gate'].ndim == 0, 'Expected scalar reference and gate')
    need(all(bool(torch.isfinite(t).all()) for t in result.values()), 'Non-finite frozen state')
    need(bool((result['mass'] >= 0).all()) and abs(float(result['mass'].sum()) - 1) < 1e-6,
         'Contribution weights must sum to one')
    q = result['responsibility']
    need(bool((q >= 0).all()) and bool(((q.sum(-1) - 1).abs() < 1e-6).all()),
         'Responsibility rows must sum to one')
    fraction(float(result['prediction']))
    need(abs(float(result['gate'])) <= 1, 'Gate outside [-1, 1]')
    return result


def local_labels(value, n):
    if value is None:
        return None
    need(set(value) == {'mask', 'responsibility'}, 'Unexpected local-label field')
    need(len(value['mask']) == len(value['responsibility']) == n, 'Local-label length mismatch')
    need(all(type(x) is bool for x in value['mask']), 'Mask must contain booleans')
    for row in value['responsibility']:
        need(len(row) == 3, 'Expected three role probabilities')
        for x in row:
            fraction(x)
        need(abs(sum(row) - 1) < 1e-6, 'Local-label probabilities must sum to one')
    return value


def training_rows(data):
    need(set(data) == {'kind', 'records'} and data['kind'] in ('training', 'synthetic'),
         'Expected a training bundle')
    result = []
    for row in data['records']:
        need(set(row) == {'id', 'target', 'clean', 'predicted', 'supervision'}, 'Training row fields')
        clean = state(row['clean'])
        pred = state(row['predicted']) if row['predicted'] is not None else None
        sup = row['supervision']
        need(set(sup) == {'clean', 'predicted'}, 'Supervision views')
        local_labels(sup['clean'], len(clean['mass']))
        if pred is None:
            need(sup['predicted'] is None, 'Blocked predicted view cannot have local supervision')
        else:
            local_labels(sup['predicted'], len(pred['mass']))
        result.append(dict(id=row['id'], target=fraction(row['target']), clean=clean,
                           predicted=pred, supervision=sup))
    ids = [r['id'] for r in result]
    need(ids and all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids),
         'Training identities must be nonempty and unique')
    return result
