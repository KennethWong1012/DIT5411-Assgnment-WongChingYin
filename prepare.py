import argparse
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


def prepare_archive(source):
    with zipfile.ZipFile(source) as archive:
        parts = sorted(name for name in archive.namelist() if re.search(r'/all_data\.zip\.\d+$', name))
        if not parts:
            return source
        combined = source.with_name('all_data.zip')
        expected_bytes = sum(archive.getinfo(name).file_size for name in parts)
        if combined.exists() and combined.stat().st_size == expected_bytes:
            return combined
        temporary = combined.with_suffix('.partial')
        with temporary.open('wb') as output:
            for name in parts:
                with archive.open(name) as part:
                    shutil.copyfileobj(part, output, 1024 * 1024)
        print(f'Combined {len(parts)} archive parts.', flush=True)
        temporary.replace(combined)
        return combined


def sample_key(name):
    return [int(piece) if piece.isdigit() else piece.casefold() for piece in re.split(r'(\d+)', name)]


def prepare(source, output, size, expected_classes):
    source = prepare_archive(source)
    output.mkdir(parents=True, exist_ok=True)
    try:
        archive_file = zipfile.ZipFile(source, metadata_encoding='big5')
    except UnicodeDecodeError:
        archive_file = zipfile.ZipFile(source, metadata_encoding='cp950')
    with archive_file as archive:
        classes = defaultdict(list)
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'} and '__MACOSX' not in path.parts:
                classes[path.parent.name].append(name)
        labels = sorted(classes, key=sample_key)
        if len(labels) != expected_classes:
            raise ValueError(f'Expected {expected_classes} classes, found {len(labels)}. Examples: {labels[:12]}')

        train, test, label_map = [], [], []
        for label_id, label in enumerate(labels):
            names = sorted(classes[label], key=lambda name: (sample_key(PurePosixPath(name).name), name))
            characters = {PurePosixPath(name).stem.rpartition('_')[0] for name in names}
            character = characters.pop() if len(characters) == 1 else ''
            if label.isdecimal() and len(character) > 1:
                # Two legacy filenames append or prepend an ASCII letter to their character.
                character = character.strip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            if label.isdecimal() and len(character) != 1:
                raise ValueError(f'Could not recover one character for class folder {label}.')
            label_map.append({'class_folder': label, 'character': character or label})
            train.extend((name, label_id) for name in names[:40])
            test.extend((name, label_id) for name in names[40:])
        if not test:
            raise ValueError('The dataset has no held-out images.')

        short = {label: len(classes[label]) for label in labels if len(classes[label]) < 40}
        no_test = sum(len(classes[label]) <= 40 for label in labels)
        fingerprint = hashlib.sha256()
        for name, _ in train + test:
            fingerprint.update(f'{name}\t{archive.getinfo(name).CRC}\n'.encode())
        training_entries = {(label, archive.getinfo(name).CRC, archive.getinfo(name).file_size) for name, label in train}
        repeated_test = sum((label, archive.getinfo(name).CRC, archive.getinfo(name).file_size) in training_entries
                            for name, label in test)
        metadata = dict(classes=len(labels), train=len(train), test=len(test), image_size=size,
                        short_training_classes=short, classes_without_test=no_test,
                        matching_test_file_checksums=repeated_test,
                        fingerprint=fingerprint.hexdigest())
        print(json.dumps(metadata, ensure_ascii=True), flush=True)

        def decode(payload):
            with Image.open(io.BytesIO(payload)) as image:
                return np.asarray(image.convert('L').resize((size, size), Image.Resampling.LANCZOS))

        for split, entries in [('train', train), ('test', test)]:
            images = np.lib.format.open_memmap(output / f'{split}_images.partial.npy', mode='w+',
                                               dtype=np.uint8, shape=(len(entries), size, size, 1))
            with ThreadPoolExecutor(max_workers=8) as pool:
                for start in range(0, len(entries), 512):
                    paths = entries[start:start + 512]
                    payloads = [archive.read(name) for name, _ in paths]
                    images[start:start + len(paths), :, :, 0] = np.stack(list(pool.map(decode, payloads)))
                    if start % 51200 == 0:
                        print(f'{split}: {start}/{len(entries)}', flush=True)
            images.flush()
            del images
            (output / f'{split}_images.partial.npy').replace(output / f'{split}_images.npy')
            np.save(output / f'{split}_labels.npy', np.array([label for _, label in entries], dtype=np.int32))
        (output / 'labels.json').write_text(json.dumps(label_map, ensure_ascii=False), encoding='utf-8')
        (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare each class in natural sample filename order.')
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', default=Path('data/prepared'), type=Path)
    parser.add_argument('--size', default=48, type=int)
    parser.add_argument('--classes', default=13065, type=int)
    args = parser.parse_args()
    prepare(args.archive, args.output, args.size, args.classes)
