"""Recorded-group split rule; no dataset or real membership is bundled.

Apply to the original 1,085-unit non-test pool before the four KU exclusions.
Supply local JSONL records with metadata.subject and metadata.student_id.
This script returns the pre-admission 927/158 assignment for the original data;
925/156 additionally requires the original knowledge-admission filter.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SALT = 'C_COMMON_20261014_V1|'

def development_groups(records):
    groups = defaultdict(set)
    for row in records:
        metadata = row['metadata']
        subject = metadata['subject']
        group = subject + '::' + metadata['student_id']
        groups[subject].add(group)
    selected = set()
    for subject, members in groups.items():
        count = min(len(members)-1, max(1, round(len(members)*.15))) if len(members)>1 else 0
        ordered = sorted(members, key=lambda group: hashlib.sha256((SALT+group).encode()).hexdigest())
        selected.update(ordered[:count])
    return selected

def assignments(records):
    selected = development_groups(records)
    for row in records:
        metadata = row['metadata']
        group = metadata['subject']+'::'+metadata['student_id']
        yield {'unit_id': row['unit_id'], 'split': 'development' if group in selected else 'train'}

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path,help='Local private membership output; do not commit it')
    args=parser.parse_args()
    records=[json.loads(line) for line in args.input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    with args.output.open('x',encoding='utf8') as stream:
        for item in assignments(records):stream.write(json.dumps(item,ensure_ascii=False)+'\n')
