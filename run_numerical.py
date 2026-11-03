"""Numerical readout from prepared prompts and a Qwen checkpoint or LoRA adapter."""
import argparse
import hashlib
import json
from pathlib import Path
from erbc_io import need, read, write


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def infer(data, model_path, adapter=None, device='cuda:0'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    need(set(data) == {'expected_ids', 'records'}, 'Input bundle fields')
    ids = data['expected_ids']
    need(ids and len(set(ids)) == len(ids) and len(data['records']) == len(ids)
         and {r['id'] for r in data['records']} == set(ids), 'Input membership mismatch')
    for r in data['records']:
        need(set(r) == {'id', 'status', 'input_ids'}, 'Model inputs cannot contain targets')
        need(r['status'] in ('valid', 'blocked'), 'Unfinished execution is not a blocked input')
        if r['status'] == 'valid':
            need(r['input_ids'] and all(type(t) is int and t >= 0 for t in r['input_ids']),
                 'Invalid input token sequence')
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    digit_ids = [tok.encode(str(i), add_special_tokens=False) for i in range(10)]
    need(all(len(x) == 1 for x in digit_ids), 'Expected ten single-token numeric anchors')
    digit_ids = [x[0] for x in digit_ids]
    need(len(set(digit_ids)) == 10, 'Numeric anchor tokens are not unique')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    model = model.to(device).eval()
    results = []
    with torch.inference_mode():
        for row in data['records']:
            if row['status'] == 'blocked':
                results.append({'id': row['id'], 'status': 'blocked'})
                continue
            x = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
            logits = model(input_ids=x, attention_mask=torch.ones_like(x),
                           use_cache=False, logits_to_keep=1).logits[:, -1, digit_ids].float()
            p = logits.softmax(-1)
            need(bool(torch.isfinite(p).all()), 'Non-finite numerical probabilities')
            value = float((p * torch.linspace(0, 1, 10, device=device)).sum())
            results.append({'id': row['id'], 'status': 'valid', 'prediction': value,
                            'token_sha256': token_hash(row['input_ids'])})
    return {'expected_ids': ids, 'records': results}


def ensemble(left, right):
    need(left['expected_ids'] == right['expected_ids'], 'Different ensemble membership')
    ids = left['expected_ids']
    a, b = ({r['id']: r for r in x['records']} for x in (left, right))
    need(len(a) == len(left['records']) == len(b) == len(right['records']) == len(ids)
         and set(a) == set(b) == set(ids), 'Missing or duplicate ensemble predictions')
    result = []
    for u in ids:
        need(a[u]['status'] == b[u]['status'], 'Different branch status')
        if a[u]['status'] == 'blocked':
            result.append({'id': u, 'status': 'blocked'})
        else:
            need(a[u]['status'] == 'valid' and a[u]['token_sha256'] == b[u]['token_sha256'],
                 'Ensemble branches must use identical tokens')
            from erbc_io import fraction
            value = .5 * fraction(a[u]['prediction']) + .5 * fraction(b[u]['prediction'])
            result.append({'id': u, 'status': 'valid', 'prediction': value,
                           'token_sha256': a[u]['token_sha256']})
    return {'expected_ids': ids, 'records': result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('infer')
    run.add_argument('--input', required=True)
    run.add_argument('--model', required=True)
    run.add_argument('--adapter')
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--output', required=True)
    avg = sub.add_parser('ensemble')
    avg.add_argument('--left', required=True)
    avg.add_argument('--right', required=True)
    avg.add_argument('--output', required=True)
    a = p.parse_args()
    if a.command == 'infer':
        result = infer(read(a.input), a.model, a.adapter, a.device)
    else:
        result = ensemble(read(a.left), read(a.right))
    write(a.output, result)
