"""Build a local CIFAR-10 copy (npz) from the official archive.

The course requires the *official* 50 000 / 10 000 CIFAR-10 split.  We keep the
images and labels byte-for-byte identical to the official release and only
re-pack them as ``.npz`` for fast loading:

    python tools/prepare_cifar10.py --parquet-dir .pkg --out data/cifar10

Input: ``train-00000-of-00001.parquet`` / ``test-00000-of-00001.parquet`` from
the ``uoft-cs/cifar10`` mirror of the official archive (60 000 32x32 RGB images,
labels 0-9 in the official order).  If you already have
``cifar-10-python.tar.gz``, ``--targz`` can be used instead; the extracted
``cifar-10-batches-py`` files are read directly.
"""
import argparse
import io
import os
import pickle
import tarfile

import numpy as np


def _from_targz(path: str):
    with tarfile.open(path, "r:gz") as tf:
        train_x, train_y, test_x, test_y = [], [], [], []
        for i in range(1, 6):
            f = tf.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            d = pickle.load(io.BytesIO(f.read()), encoding="latin1")
            arr = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            train_x.append(arr)
            train_y.append(np.array(d["labels"]))
        f = tf.extractfile("cifar-10-batches-py/test_batch")
        d = pickle.load(io.BytesIO(f.read()), encoding="latin1")
        test_x.append(d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        test_y.append(np.array(d["labels"]))
    return (np.concatenate(train_x), np.concatenate(train_y),
            np.concatenate(test_x), np.concatenate(test_y))


def _from_parquet(dirname: str):
    import pyarrow.parquet as pq
    from PIL import Image

    def _read(split):
        t = pq.read_table(os.path.join(dirname, f"{split}-00000-of-00001.parquet"))
        imgs, labels = [], []
        for row in t.to_pylist():
            imgs.append(np.array(Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")))
            labels.append(row["label"])
        return np.stack(imgs), np.array(labels, dtype=np.int64)

    tr_x, tr_y = _read("train")
    te_x, te_y = _read("test")
    return tr_x, tr_y, te_x, te_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default=".pkg")
    p.add_argument("--targz", default=None)
    p.add_argument("--out", default="data/cifar10")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.targz:
        tr_x, tr_y, te_x, te_y = _from_targz(args.targz)
    else:
        tr_x, tr_y, te_x, te_y = _from_parquet(args.parquet_dir)

    # torch expects NCHW; store uint8 to keep the file small (150 MB total)
    tr_x = np.transpose(tr_x, (0, 3, 1, 2)).astype(np.uint8)
    te_x = np.transpose(te_x, (0, 3, 1, 2)).astype(np.uint8)
    np.savez(os.path.join(args.out, "train.npz"), x=tr_x, y=tr_y.astype(np.int64))
    np.savez(os.path.join(args.out, "test.npz"), x=te_x, y=te_y.astype(np.int64))

    for name, x, y in (("train", tr_x, tr_y), ("test", te_x, te_y)):
        counts = np.bincount(y, minlength=10)
        print(f"{name}: {x.shape} labels per class {counts.tolist()}")
    assert tr_x.shape[0] == 50000 and te_x.shape[0] == 10000, "official split violated"


if __name__ == "__main__":
    main()

