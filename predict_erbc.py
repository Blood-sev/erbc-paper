"""Apply a trained final head to label-free frozen states."""
import argparse
from erbc_io import ARMS, need, read, write, state


def predict(data, checkpoint=None, arm='joint'):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from erbc_head import BiasReadout
    need(set(data) == {'expected_ids', 'records'}, 'Unexpected inference bundle fields')
    ids = data['expected_ids']
    need(ids and len(set(ids)) == len(ids), 'Empty or duplicate expected identities')
    need(len(data['records']) == len(ids) and {r['id'] for r in data['records']} == set(ids),
         'Incomplete or duplicate inference records')
    need(arm in ARMS, 'Unknown correction condition')
    head = BiasReadout(arm).eval()
    if checkpoint:
        with safe_open(str(checkpoint), framework='pt', device='cpu') as f:
            meta = f.metadata() or {}
        need(meta.get('arm', arm) == arm, 'Checkpoint condition mismatch')
        head.load_state_dict(load_file(str(checkpoint)), strict=True)
    output = []
    with torch.no_grad():
        for row in data['records']:
            need(set(row) == {'id', 'status', 'state'}, 'Inference rows cannot contain labels')
            if row['status'] == 'blocked':
                need(row['state'] is None, 'Blocked record must not carry a state')
                output.append({'id': row['id'], 'status': 'blocked'})
                continue
            need(row['status'] == 'valid', 'Execution failures cannot be scored')
            z = head(state(row['state']))
            output.append({'id': row['id'], 'status': 'valid', 'prediction': float(z['prediction']),
                           'delta_q': float(z['delta_q']), 'delta_b': float(z['delta_bias']),
                           'responsibility': z['responsibility'].tolist()})
    return {'expected_ids': ids, 'records': output}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--frozen-reference', action='store_true')
    p.add_argument('--arm', choices=ARMS, default='joint')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    write(a.output, predict(read(a.input), a.checkpoint, a.arm))
