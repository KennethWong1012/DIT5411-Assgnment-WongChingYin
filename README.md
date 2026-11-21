# Traditional Chinese Handwriting Recognition

This DIT5411 project compares three TensorFlow classifiers on 13,065 classes of handwritten traditional Chinese characters. Each classifier predicts one label per grayscale image.

## Dataset and split

Use the [full Traditional Chinese Handwriting Dataset](https://github.com/chenkenanalytic/handwritting_data_all). Within each character folder the script sorts filenames in natural numeric order, assigns the first 40 original images to training, and holds out every later image for testing. The script records the classes with fewer original images, the number of classes without held-out images and test files whose ZIP checksums match a training file in the same class. The saved label map connects output indices to character names decoded from the original Big5 filenames.

Images are resized to 48 × 48 pixels and normalized with a dark background. Each epoch uses 200 examples per class. For classes with 40 originals, the training stream includes each original once and four transformed versions. Labels with fewer originals receive additional variants. Each transform combines small rotation (up to 8°), shear (up to 0.10), and scaling (up to about 12%). The transforms are deterministic, use bilinear interpolation, and are computed in memory. The test set receives the same grayscale preprocessing and no geometric augmentation.

## Models

| Model | Architecture |
| --- | --- |
| Shallow CNN | Two convolutional stages with max pooling |
| Deep CNN | Three stages with additional separable convolutions |
| Residual CNN | Three stages with skip connections around separable convolutions |

The three models share a 192-unit dense layer with batch normalization and a leaky ReLU activation, followed by a 13,065-unit classification layer. They use Adam and sparse categorical cross-entropy on logits. Top-1 accuracy counts an image as correct when the character with the largest logit is its label. The comparison uses the same test set and, where all models finish, the same number of complete epochs.

## Reproduce the experiment

Use Python 3.12. On Windows, native TensorFlow runs on CPU. For NVIDIA GPU training, use TensorFlow under WSL2 or Linux and install the CUDA dependencies with `python -m pip install "tensorflow[and-cuda]==2.21.0"`.

```sh
python -m pip install -r requirements.txt
```

Download the [full dataset ZIP](https://codeload.github.com/chenkenanalytic/handwritting_data_all/zip/refs/heads/master) as `data/upstream.zip`. `prepare.py` also accepts the combined `all_data.zip` and reads the inner ZIP directly.

```sh
python prepare.py data/upstream.zip
python train.py --benchmark 40 --output run/benchmark
python train.py --batch 256 --epochs 4 --hours 8
```

Run the training command again with the same settings to continue from the complete checkpoints under `run/checkpoints`. Change `--hours` or increase `--epochs` to extend the run. The run records the split fingerprint with the training configuration. Measurements are written to `run/results.csv` and `run/RESULTS.md`.

To recognize a single image, choose a completed checkpoint and run `python predict.py sample.png run/checkpoints/shallow-1.keras`.

## Held-out test results

<!-- results:start -->

The source has 13,065 classes, 522,127 original training images and 162,550 test images. 43 classes have no held-out images. The ZIP checksum audit found 0 test files that also match a training file in the same class.

| Model | Epochs | Top-1 accuracy |
| --- | ---: | ---: |
| deep | 4 | 92.06% |
| residual | 4 | 92.00% |
| shallow | 4 | 89.20% |

deep achieved the highest held-out Top-1 accuracy in this comparison.

<!-- results:end -->

## References

- [AI . FREE Team Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset), adapted from STUST EECS Chinese MNIST and originally produced from Tegaki.
- [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV) for background on image transformation.
- [Handwritten Digit Recognition using Keras and TensorFlow](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow) for background on CNN-based handwritten recognition.

The project code is under the [MIT License](LICENSE). The upstream dataset is distributed separately under the [CC BY-NC-SA 4.0 terms stated by its authors](https://github.com/chenkenanalytic/handwritting_data_all#license-授權).
