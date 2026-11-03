"""Verify the release files against SHA256SUMS."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
count = 0
for line in (root / 'SHA256SUMS').read_text(encoding='utf8').splitlines():
    digest, name = line.split('  ', 1)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Unsafe manifest path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Checksum mismatch: ' + name)
    count += 1
print(f'PASS: {count} release files verified')

import ast
import json
from zipfile import ZipFile
from pathlib import PurePosixPath
for archive in root.glob('*.zip'):
    with ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Invalid archive: ' + archive.name)
        for name in z.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive path')
            if name.endswith('.py'):
                ast.parse(z.read(name).decode('utf-8-sig'), filename=name)
with ZipFile(root / 'erbc_source_snapshot.zip') as z:
    entries = json.loads((root / 'SOURCE_INDEX.json').read_text(encoding='utf8'))['files']
    if {e['path'] for e in entries} != set(z.namelist()):
        raise ValueError('Source index coverage mismatch')
    for entry in entries:
        if hashlib.sha256(z.read(entry['path'])).hexdigest() != entry['sha256']:
            raise ValueError('Source hash mismatch')
print('PASS: source index, archive integrity, and Python syntax')
