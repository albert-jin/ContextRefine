"""Verify release files and explicitly account for documented report localization."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    localization = json.loads((ROOT / 'LOCALIZATION.json').read_text(encoding='utf-8'))
    changes = {row['path']: row for row in localization['files']}
    for row in manifest['files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Release file mismatch: ' + row['path'])
    for path, change in changes.items():
        if sha(ROOT / path) != change['localized_sha256']:
            raise ValueError('Localized file mismatch: ' + path)
    localized_matches = 0
    for run in manifest['runs']:
        repo = ROOT / run['project']
        saved = repo / 'auto_res_logs' / 'runs' / run['run'] / 'source_hashes.json'
        for path, expected in json.loads(saved.read_text(encoding='utf-8')).items():
            actual = sha(repo / path)
            if actual == expected:
                continue
            key = (Path(run['project']) / path).as_posix()
            change = changes.get(key)
            if not change or expected != change['archived_sha256'] or actual != change['localized_sha256']:
                raise ValueError('Undocumented archived-source difference: ' + key)
            localized_matches += 1
    print('PASS: release file hashes match.')
    print('PASS: archived run inventories match except for {} documented report-localization entries.'.format(localized_matches))
    print('Historical test locks are unchanged; exact replay still requires original archived source bytes.')


if __name__ == '__main__':
    main()
