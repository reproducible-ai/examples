#!/usr/bin/env python3
"""Fetch the content-image dataset this row trains on.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
`fast_neural_style/neural_style/neural_style.py train` takes `--dataset <dir>` and
feeds it to `torchvision.datasets.ImageFolder`. Upstream ships no downloader: the
README points a human at the COCO 2014 *training* split ("80K/13GB") and stops
there. A recorded pipeline cannot start from a manual browser download, so the
fetch has to be a real, traced, first step -- otherwise the training step's input
appears in the lineage from nowhere.

WHY IMAGENETTE AND NOT COCO
---------------------------
COCO 2014 train is 13 GB / 82,783 images. Style transfer's loss is content-agnostic
(a perceptual loss against VGG-16 features plus a Gram-matrix style term); nothing
in the objective depends on COCO's annotations, its object categories, or its
particular photographs -- only on a large, varied corpus of natural images. So the
dataset is a *scale* choice, not a semantic one, which is exactly the kind of
substitution a truncated reproduction is allowed to make as long as it is declared.
Imagenette (a 10-class subset of ImageNet, the 320px edition, ~325 MB / 9,469 train
images) is natural photography of the same character, is a single public tarball
with no credentials, and downloads in well under a minute instead of tens of
minutes. The row is truncated on TWO axes and both are stated in row.json: 1 of
upstream's default 2 epochs, over 9,469 images rather than 82,783.

ImageFolder needs `<root>/<class>/<image>`; imagenette2-320/train already has that
shape (10 wnid subdirectories), so the archive is used exactly as distributed --
no re-layout, no copying, no symlinking. That matters beyond convenience: copying
or hard-linking image files would duplicate bytes at two paths, which is the
finalize-time hazard this campaign screens every row for.

Standard library only, deliberately. urllib + tarfile means this step adds no
dependency that the training step does not already need, so the recorded package
set stays an honest description of the workload rather than of the fetcher.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.request

DEFAULT_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def download(url: str, dest: str) -> None:
    t0 = time.time()
    # Reported by the *server*, not trusted for correctness -- the sha256 below is
    # what establishes identity. This is only so the log shows progress honestly.
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total and got % (32 << 20) < (1 << 20):
                print(f"  {got / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    print(f"downloaded {got} bytes in {time.time() - t0:.1f}s", flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--expect-train-images",
        type=int,
        default=None,
        help="assert the extracted train split has exactly this many images; the "
        "training step's checkpoint filename is derived from the batch count, so "
        "a different count would silently produce a differently-named artifact",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    archive = os.path.join(args.out_dir, os.path.basename(args.url))
    root = os.path.join(args.out_dir, "imagenette2-320")

    if not os.path.isdir(root):
        if not os.path.exists(archive):
            print(f"fetching {args.url}", flush=True)
            download(args.url, archive)
        print(f"archive sha256 {sha256(archive)}", flush=True)
        t0 = time.time()
        with tarfile.open(archive) as tf:
            # `filter="data"` is the safe extraction policy Python 3.12 warns about
            # when omitted; it refuses absolute paths, parent-directory escapes,
            # links and device nodes. Nothing in this archive needs the old
            # behaviour, and the refusal to extract *links* is a bonus here: a
            # symlinked duplicate would be exactly the byte-duplication hazard the
            # campaign screens for.
            tf.extractall(args.out_dir, filter="data")
        print(f"extracted in {time.time() - t0:.1f}s", flush=True)
    else:
        print(f"{root} already present, skipping download", flush=True)

    counts = {}
    for split in ("train", "val"):
        split_dir = os.path.join(root, split)
        n = sum(
            1
            for d, _, fs in os.walk(split_dir)
            for f in fs
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        classes = sorted(
            e for e in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, e))
        )
        counts[split] = n
        print(f"{split}: {n} images across {len(classes)} classes", flush=True)

    if args.expect_train_images is not None and counts["train"] != args.expect_train_images:
        print(
            f"ERROR: expected {args.expect_train_images} train images, found "
            f"{counts['train']}. The training step's --checkpoint-interval is set "
            f"from this number; stopping before spending GPU time.",
            file=sys.stderr,
        )
        return 1

    print(f"dataset root for --dataset: {os.path.join(root, 'train')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
