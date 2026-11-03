"""Score a fixed evaluation collection in percentage points; standard library only."""
import argparse
import json
import math
from pathlib import Path


def unit_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected a numeric value')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a finite value in [0, 1]')
    return value


def score(payload):
    expected = payload['expected_ids']
    records = payload['records']
    if not expected or any(not isinstance(i, str) or not i for i in expected):
        raise ValueError('A nonempty fixed membership list is required')
    ids = [r['id'] for r in records]
    if len(set(expected)) != len(expected) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate identity')
    if set(ids) != set(expected):
        raise ValueError('Missing or unexpected evaluation identity')
    errors, valid = [], []
    counts = dict(valid=0, blocked=0, invalid=0)
    for row in records:
        target = unit_value(row['target'])
        status = row['status']
        if status not in counts:
            raise ValueError('Unfinished execution or unsupported status: ' + str(status))
        counts[status] += 1
        error = 1.0
        if status == 'valid':
            error = abs(unit_value(row['prediction']) - target)
            valid.append(error)
        errors.append(error)

    def metrics(values):
        if not values:
            return None, None
        return (100 * math.fsum(values) / len(values),
                100 * math.sqrt(math.fsum(e * e for e in values) / len(values)))

    mae, rmse = metrics(errors)
    vmae, vrmse = metrics(valid)
    return dict(total=len(expected), counts=counts, pMAE=mae, pRMSE=rmse,
                valid_MAE=vmae, valid_RMSE=vrmse, unit='percentage points')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.input.read_text(encoding='utf-8-sig'))), indent=2, allow_nan=False))
