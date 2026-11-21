import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from train import normalize


def predict(image_path, checkpoint, labels_path, count):
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    model = tf.keras.models.load_model(checkpoint, compile=False)
    if len(labels) != model.output_shape[-1]:
        raise ValueError('The label map does not match the checkpoint.')
    _, height, width, _ = model.input_shape
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert('L').resize((width, height), Image.Resampling.LANCZOS))
    logits = model(normalize(gray[None, :, :, None]), training=False)
    best = tf.math.top_k(tf.nn.softmax(logits)[0], k=min(count, len(labels)))
    for score, index in zip(best.values.numpy(), best.indices.numpy()):
        label = labels[index]
        print(f"{label['character']}\t{label['class_folder']}\t{score:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rank candidate Traditional Chinese characters for one image.')
    parser.add_argument('image', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--labels', default=Path('data/prepared/labels.json'), type=Path)
    parser.add_argument('--top', default=5, type=int)
    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.labels, args.top)
