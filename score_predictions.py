"""Join completed predictions with labels by exact evaluation identity."""
import argparse
from erbc_io import need, read, write
from evaluate import score as evaluate


def score(predictions, labels):
    ids = predictions['expected_ids']
    gold = {r['id']: r['target'] for r in labels['records']}
    need(len(gold) == len(labels['records']) and set(gold) == set(ids), 'Label membership mismatch')
    records = []
    for row in predictions['records']:
        r = {'id': row['id'], 'status': row['status'], 'target': gold[row['id']]}
        if row['status'] == 'valid':
            r['prediction'] = row['prediction']
        records.append(r)
    return evaluate({'expected_ids': ids, 'records': records})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    write(a.output, score(read(a.predictions), read(a.labels)))
