"""One-time, byte-preserving comparison rename and mechanical reference migration."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'benchmarks/comparison-video-renames.json'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    mapping = json.loads(MANIFEST.read_text())
    pairs = [(ROOT / old, ROOT / new) for old, new in mapping.items()]
    assert len(set(mapping.values())) == len(mapping)
    for old, new in pairs:
        assert old.is_file(), old
        assert not new.exists(), new
    hashes = {str(new.relative_to(ROOT)): digest(old) for old, new in pairs}
    changed = []
    extensions = {'.md', '.html', '.json', '.py', '.ipynb', '.txt', '.sh'}
    for folder in ('benchmarks', 'docs', 'scripts'):
        for path in (ROOT / folder).rglob('*'):
            if not path.is_file() or path.suffix not in extensions:
                continue
            if path in (MANIFEST, Path(__file__).resolve()):
                continue
            try:
                before = path.read_text()
            except UnicodeError:
                continue
            replacements = {}
            for old, new in pairs:
                replacements[str(old)] = str(new)
                replacements[str(old.relative_to(ROOT))] = str(new.relative_to(ROOT))
                rel_old = os.path.relpath(old, path.parent)
                rel_new = os.path.relpath(new, path.parent)
                # Relative link with sufficient directory context.
                if '/' in rel_old or path.parent == old.parent:
                    replacements[rel_old] = rel_new
                # Files inside the same experiment may use local root-relative paths.
                if path.is_relative_to(old.parent):
                    replacements[old.name] = new.name
                for ancestor in old.parents:
                    if ancestor == ROOT:
                        break
                    if ancestor != old.parent:
                        fragment = str(old.relative_to(ancestor))
                        if '/' in fragment:
                            replacements[fragment] = str(new.relative_to(ancestor))
            after = before
            for old, new in sorted(replacements.items(), key=lambda row: -len(row[0])):
                after = after.replace(old, new)
            if after != before:
                if path.suffix in ('.json', '.ipynb'):
                    json.loads(after)
                path.write_text(after)
                changed.append(str(path.relative_to(ROOT)))
    for old, new in pairs:
        old.rename(new)
        assert digest(new) == hashes[str(new.relative_to(ROOT))], new
    audit = {'execution': 'CPU/filesystem-only rename; no GPU inference or re-encoding',
             'renamed_count': len(pairs), 'sha256_after_verified_equal_to_before': hashes,
             'updated_text_files': changed}
    (ROOT / 'benchmarks/comparison-video-rename-validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'renamed': len(pairs), 'references_updated_in_files': len(changed)}))


if __name__ == '__main__':
    main()
