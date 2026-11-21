import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODELS = {
    'shallow': 'Two convolutional stages with max pooling',
    'deep': 'Three stages with additional convolutions',
    'residual': 'Three stages with residual separable-convolution blocks',
}


def conv(x, width, stride=1, separable=False, activate=True):
    layer = layers.SeparableConv2D if separable else layers.Conv2D
    x = layer(width, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x) if activate else x


def make_model(name, size, classes):
    inputs = keras.Input((size, size, 1))
    x = conv(inputs, 32, stride=2)
    for width in (32, 64) if name == 'shallow' else (32, 64, 96):
        x = conv(x, width)
        if name == 'deep':
            x = conv(x, width, separable=True)
        if name == 'residual':
            branch = conv(x, width, separable=True)
            branch = conv(branch, width, separable=True, activate=False)
            x = layers.Activation('relu')(layers.Add()([x, branch]))
        if x.shape[1] > 6:
            x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(192, use_bias=False)(x)
    x = layers.BatchNormalization(name='feature_normalization')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='character_features')(x)
    logits = layers.Dense(classes, dtype='float32', name='character_logits')(x)
    model = keras.Model(inputs, logits, name=name)
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='top1')], jit_compile=False)
    return model


def normalize(images):
    images = tf.cast(images, tf.float32) / 255.0
    corners = (images[:, 0, 0] + images[:, -1, 0] + images[:, 0, -1] + images[:, -1, -1]) / 4
    return tf.where(corners[:, None, None, :] > 0.5, 1.0 - images, images)


def augment(images, ids, size):
    # Separate deterministic random stream per sample and variant. ID zero modulo five is the original.
    def draw(salt, magnitude):
        bits = tf.math.floormod(tf.cast(ids, tf.int64) * tf.constant(salt, tf.int64) + 17713, 104729)
        unit = tf.cast(bits, tf.float32) / 104728.0
        return tf.where(ids % 5 == 0, 0.0, (2 * unit - 1) * magnitude)

    angle = draw(27347, 0.14)
    shear = draw(51631, 0.10)
    scale = 1 + draw(39199, 0.12)
    a = tf.cos(angle) / scale
    b = (tf.sin(angle) + shear) / scale
    c = -tf.sin(angle) / scale
    d = tf.cos(angle) / scale
    center = (size - 1) / 2
    zeros = tf.zeros_like(a)
    transforms = tf.stack([a, b, center * (1 - a - b), c, d, center * (1 - c - d), zeros, zeros], axis=1)
    return tf.raw_ops.ImageProjectiveTransformV3(images=images, transforms=transforms,
                                                output_shape=[size, size], interpolation='BILINEAR',
                                                fill_mode='CONSTANT', fill_value=0.0)


class Data:
    def __init__(self, prepared, batch):
        self.meta = json.loads((prepared / 'metadata.json').read_text(encoding='utf-8'))
        self.size = self.meta['image_size']
        self.batch = batch
        self.train_labels = np.load(prepared / 'train_labels.npy')
        self.test_labels = np.load(prepared / 'test_labels.npy')
        ids = np.arange(len(self.train_labels), dtype=np.int32) * 5
        counts = np.bincount(self.train_labels, minlength=self.meta['classes'])
        if not counts.min() or counts.max() > 40:
            raise ValueError('Prepared labels violate the first-40 split.')
        # Ensure 200 examples even if the source has fewer than 40 original images for a label.
        starts = np.cumsum(counts) - counts
        if np.any(counts < 40):
            self.rows = np.concatenate([starts[label] + np.arange(200) % count
                                        for label, count in enumerate(counts)]).astype(np.int32)
            self.ids = np.concatenate([
                np.r_[ids[start:start+count], np.full(200-count, -1)]
                for start, count in zip(starts, counts)
            ]).astype(np.int32)
            # First count examples are originals. Remaining IDs select unique transforms.
            modified = self.ids == -1
            self.ids[modified] = np.arange(np.count_nonzero(modified), dtype=np.int32) // 4 * 5 + 1 + np.arange(np.count_nonzero(modified), dtype=np.int32) % 4
        else:
            self.rows = np.repeat(np.arange(len(ids), dtype=np.int32), 5)
            self.ids = np.arange(len(self.rows), dtype=np.int32)
        with tf.device('/CPU:0'):
            self.train_images = tf.convert_to_tensor(np.load(prepared / 'train_images.npy', mmap_mode='r'))
            self.test_images = tf.convert_to_tensor(np.load(prepared / 'test_images.npy', mmap_mode='r'))
            self.train_labels = tf.convert_to_tensor(self.train_labels)

    def training(self, epoch, seed, limit=None):
        order = np.random.default_rng(seed + epoch).permutation(len(self.rows))
        if limit:
            order = order[:limit * self.batch]
        stream = tf.data.Dataset.from_tensor_slices((self.rows[order], self.ids[order])).batch(self.batch)

        def transform(rows, ids):
            with tf.device('/CPU:0'):
                return augment(normalize(tf.gather(self.train_images, rows)), ids, self.size), tf.gather(self.train_labels, rows)

        return stream.map(transform, num_parallel_calls=4).prefetch(2)

    def testing(self):
        stream = tf.data.Dataset.from_tensor_slices((tf.range(len(self.test_labels)), self.test_labels)).batch(self.batch)

        def read(rows, labels):
            with tf.device('/CPU:0'):
                return normalize(tf.gather(self.test_images, rows)), labels

        return stream.map(read, num_parallel_calls=2).prefetch(2)


def atomic_text(path, content):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(content, encoding='utf-8')
    temp.replace(path)


def completed_epochs(directory, name):
    return sorted(int(path.stem.rsplit('-', 1)[1]) for path in directory.glob(f'{name}-[0-9]*.keras'))


def report(directory, rows, meta):
    rows = sorted(rows, key=lambda row: (-row['top1'], row['model']))
    with (directory / 'results.csv.partial').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'epochs', 'test_images', 'top1', 'test_loss', 'parameters'])
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'results.csv.partial').replace(directory / 'results.csv')
    lines = ['# Test results', '', f"{meta['test']:,} original images in the held-out test split.", '',
             '| Model | Epochs | Top-1 accuracy |', '| --- | ---: | ---: |']
    lines.extend(f"| {row['model']} | {row['epochs']} | {row['top1']:.2%} |" for row in rows)
    if len(rows) == len(MODELS):
        lines.extend(['', f"{rows[0]['model']} achieved the highest held-out Top-1 accuracy in this comparison."])
    atomic_text(directory / 'RESULTS.md', '\n'.join(lines) + '\n')
    if directory.resolve() == Path('run').resolve():
        readme = Path('README.md')
        text = readme.read_text(encoding='utf-8')
        before, _, tail = text.partition('<!-- results:start -->')
        _, _, after = tail.partition('<!-- results:end -->')
        if tail and after:
            details = [f"The source has {meta['classes']:,} classes, {meta['train']:,} original training images and "
                       f"{meta['test']:,} test images. {meta['classes_without_test']:,} classes have no held-out images. "
                       f"The ZIP checksum audit found {meta.get('matching_test_file_checksums', 0):,} test files that "
                       'also match a training file in the same class.', '',
                       *lines[4:]]
            atomic_text(readme, before + '<!-- results:start -->\n\n' + '\n'.join(details) + '\n\n<!-- results:end -->' + after)


class Deadline(keras.callbacks.Callback):
    def __init__(self, stop_at):
        super().__init__()
        self.stop_at = stop_at
        self.expired = False

    def on_train_batch_end(self, batch, logs=None):
        if time.monotonic() >= self.stop_at:
            self.expired = True
            self.model.stop_training = True


def run(args):
    started = time.monotonic()
    devices = tf.config.list_physical_devices('GPU')
    if args.require_gpu and not devices:
        raise RuntimeError('TensorFlow could not find a GPU.')
    if devices:
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        keras.mixed_precision.set_global_policy('mixed_float16')
    else:
        tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    print(f'Device: {devices or "CPU"}', flush=True)
    data = Data(args.data, args.batch)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    config = dict(fingerprint=data.meta['fingerprint'], size=data.size, batch=args.batch, seed=args.seed,
                  gpu=bool(devices), model_revision=2)
    if not args.benchmark:
        saved = args.output / 'config.json'
        if saved.exists() and json.loads(saved.read_text(encoding='utf-8')) != config:
            raise RuntimeError('Run settings differ from saved checkpoints. Use a fresh --output directory.')
        atomic_text(saved, json.dumps(config, indent=2))

    models = {}
    for name in MODELS:
        keras.utils.set_random_seed(args.seed)
        epochs = completed_epochs(checkpoint_dir, name) if not args.benchmark else []
        models[name] = (keras.models.load_model(checkpoint_dir / f'{name}-{epochs[-1]}.keras') if epochs
                        else make_model(name, data.size, data.meta['classes']))
        if args.benchmark:
            model = models[name]
            model.fit(data.training(0, args.seed, limit=3), verbose=0, shuffle=False)
            sample = data.training(0, args.seed, limit=args.benchmark)
            mark = time.monotonic()
            history = model.fit(sample, verbose=0, shuffle=False)
            batches = min(len(data.rows), args.batch * args.benchmark)
            print(f'{name}: {batches / (time.monotonic() - mark):,.0f} images/s; '
                  f'train top1={history.history["top1"][-1]:.4f}', flush=True)
    if args.benchmark:
        return

    deadline = started + args.hours * 3600
    # Reserve time to evaluate the small held-out set on every completed model.
    train_until = deadline - min(1800, args.hours * 360)
    interrupted = False
    for epoch in range(1, args.epochs + 1):
        for name, model in models.items():
            if epoch in completed_epochs(checkpoint_dir, name):
                continue
            if time.monotonic() >= train_until:
                interrupted = True
                break
            guard = Deadline(train_until)
            start = time.monotonic()
            print(f'{name} epoch {epoch} started', flush=True)
            history = model.fit(data.training(epoch, args.seed), callbacks=[guard], verbose=0, shuffle=False)
            if guard.expired:
                interrupted = True
                break
            pending = checkpoint_dir / f'.pending-{name}.keras'
            model.save(pending)
            pending.replace(checkpoint_dir / f'{name}-{epoch}.keras')
            print(f'{name} epoch {epoch} completed in {(time.monotonic()-start)/60:.1f} min; '
                  f'train top1={history.history["top1"][-1]:.4f}', flush=True)
        if interrupted:
            break

    latest = {name: completed_epochs(checkpoint_dir, name) for name in MODELS}
    common = min((max(epochs, default=0) for epochs in latest.values()), default=0)
    rows = []
    for name, epochs in latest.items():
        if epochs:
            epoch = common or epochs[-1]
            model = keras.models.load_model(checkpoint_dir / f'{name}-{epoch}.keras')
            metrics = model.evaluate(data.testing(), return_dict=True, verbose=0)
            rows.append(dict(model=name, epochs=epoch, test_images=data.meta['test'], top1=metrics['top1'],
                             test_loss=metrics['loss'], parameters=model.count_params()))
            report(args.output, rows, data.meta)
            print(f'{name} test top1={metrics["top1"]:.4f} after {epoch} epochs', flush=True)
    if not rows:
        raise RuntimeError('No model completed a full epoch before the training time limit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and compare three TensorFlow character classifiers.')
    parser.add_argument('--data', default=Path('data/prepared'), type=Path)
    parser.add_argument('--output', default=Path('run'), type=Path)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--hours', default=8, type=float)
    parser.add_argument('--seed', default=5411, type=int)
    parser.add_argument('--benchmark', default=0, type=int, help='Batches per model to time without saving weights.')
    parser.add_argument('--require-gpu', action='store_true')
    run(parser.parse_args())
