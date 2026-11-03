"""Verify and extract the indexed paper implementations into a working directory."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from erbc_io import need, read


def extract(destination):
    root = Path(__file__).resolve().parent
    entries = {r['path']: r['sha256'] for r in read(root / 'SOURCE_INDEX.json')['files']}
    with ZipFile(root / 'erbc_source_snapshot.zip') as archive:
        need(set(archive.namelist()) == set(entries) and len(archive.namelist()) == len(entries),
             'Archive/index membership mismatch')
        for name, digest in entries.items():
            path = PurePosixPath(name)
            need(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
                 'Unsafe archive path')
            need(hashlib.sha256(archive.read(name)).hexdigest() == digest, 'Source hash mismatch')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name in entries:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    print(f'Extracted {len(entries)} verified source files to {destination}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    extract(p.parse_args().output)
