"""Train the three-stage final correction head on frozen evidence states."""
import argparse
import copy
import hashlib
import math
import random
from pathlib import Path
from erbc_io import ARMS, need, read, write, training_rows


def objective(head, clean, pred, target, supervision, teacher_clean, teacher_pred, stage):
    from erbc_head import loss_parts
    from erbc_losses import new_loss
    if stage == 0:
        return loss_parts(head, clean, pred, target, supervision)
    _, terms = new_loss(head, clean, pred, target, supervision,
                        teacher_clean, teacher_pred, 'success_reference')
    if stage == 2:
        terms = {k: v * .25 if k in ('clean_bias', 'pred_bias') else v
                 for k, v in terms.items()}
    return sum(terms.values()), terms


def train(data_path, config_path, output):
    import torch
    from safetensors.torch import save_file
    from erbc_head import BiasReadout
    from erbc_losses import detach_output
    cfg, bundle = read(config_path), read(data_path)
    rows = training_rows(bundle)
    need(len(rows) == cfg['fit_count'], 'Fit count differs from the configuration')
    need(cfg['batch'] > 0 and cfg['threads'] > 0, 'Invalid batch/thread count')
    need(cfg['arms'] and len(set(cfg['arms'])) == len(cfg['arms']) and
         set(cfg['arms']) <= set(ARMS), 'Invalid correction conditions')
    need(cfg['seeds'] and len(set(cfg['seeds'])) == len(cfg['seeds']), 'Duplicate or empty seeds')
    need(len(cfg['stages']) == 3 and all(s['epochs'] == 4 for s in cfg['stages']),
         'The paper protocol uses three stages of four epochs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg['threads'])
    write(output / 'config.json', cfg)
    write(output / 'input_binding.json', {
        'sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'kind': bundle['kind'], 'fit_count': len(rows), 'torch': torch.__version__})
    runs = []
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            torch.manual_seed(seed)
            head = BiasReadout(arm)
            folder = output / f'{arm}_{seed}'
            folder.mkdir()
            steps = 0
            for stage, settings in enumerate(cfg['stages']):
                teacher = copy.deepcopy(head).eval().requires_grad_(False) if stage else None
                targets = []
                if teacher is not None:
                    with torch.no_grad():
                        targets = [(detach_output(teacher(r['clean'])),
                                    detach_output(teacher(r['predicted'])) if r['predicted'] is not None else None)
                                   for r in rows]
                optimizer = torch.optim.AdamW(head.parameters(), lr=settings['learning_rate'],
                                             betas=(.9, .95), eps=1e-8, weight_decay=.01)
                head.train()
                for epoch in range(1, 5):
                    order = list(range(len(rows)))
                    random.Random(seed + epoch).shuffle(order)
                    for start in range(0, len(order), cfg['batch']):
                        batch = order[start:start + cfg['batch']]
                        optimizer.zero_grad(set_to_none=True)
                        for i in batch:
                            r = rows[i]
                            tc, tp = targets[i] if targets else (None, None)
                            loss, _ = objective(head, r['clean'], r['predicted'], r['target'],
                                                r['supervision'], tc, tp, stage)
                            need(bool(torch.isfinite(loss)), 'Non-finite loss')
                            (loss / len(batch)).backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
                        optimizer.step()
                        need(all(bool(torch.isfinite(p).all()) for p in head.parameters()),
                             'Non-finite trained parameters')
                        steps += 1
                save_file(head.state_dict(), str(folder / f'stage{stage + 1}.safetensors'),
                          metadata={'arm': arm, 'seed': str(seed), 'stage': str(stage + 1)})
            expected = 12 * math.ceil(len(rows) / cfg['batch'])
            need(steps == expected, 'Unexpected update count')
            record = {'arm': arm, 'seed': seed, 'steps': steps,
                      'checkpoint': f'{arm}_{seed}/stage3.safetensors'}
            runs.append(record)
            print(f'{arm}, seed {seed}: {steps} updates', flush=True)
    write(output / 'runs.json', runs)
    return runs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='training_config.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(args.data, args.config, args.output)
