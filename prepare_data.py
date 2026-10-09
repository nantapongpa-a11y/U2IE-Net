"""Build the U2IE-Net training / test folders from the raw public datasets.

Output layout (<out>):
    train/underwater/   unpaired underwater images (UIEB raw-890 + RUIE-UIQS, minus T400)
    train/haze/         RESIDE-beta OTS hazy images
    train/clear/        matching OTS clear images (same filename as in haze/)
    test/T400/          400 held-out underwater images from the same sources
    split.json          exact file lists, so the split can be reproduced

Training images are resized to <size> and saved as PNG (lossless), which is what
train.py would do on the fly anyway; test images are copied unchanged.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import zipfile

from PIL import Image
from tqdm import tqdm

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        out += [os.path.join(dirpath, f) for f in files if f.lower().endswith(IMG_EXTS)]
    return sorted(out)


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def save_resized(img, path, size):
    img.convert('RGB').resize((size, size), Image.BILINEAR).save(path)


def build_underwater(sources, exclude_dirs, out, n_test, size, rng):
    # (prefix, path) so files from different datasets never collide by name
    pool, seen = [], set()
    for d in exclude_dirs:
        seen.update(md5(p) for p in list_images(d))
    n_dup = 0
    for prefix, d in sources:
        for p in list_images(d):
            h = md5(p)
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)
            pool.append((prefix, p))
    if n_dup:
        print(f">>> skipped {n_dup} duplicate / excluded underwater images")

    rng.shuffle(pool)
    test, train = pool[:n_test], pool[n_test:]

    test_dir = os.path.join(out, 'test', 'T400')
    train_dir = os.path.join(out, 'train', 'underwater')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    names = {'T400': [], 'underwater': []}
    for prefix, p in tqdm(test, desc='T400'):
        name = f"{prefix}_{os.path.basename(p)}"
        shutil.copy(p, os.path.join(test_dir, name))
        names['T400'].append(name)
    for prefix, p in tqdm(train, desc='underwater'):
        name = f"{prefix}_{os.path.splitext(os.path.basename(p))[0]}.png"
        with Image.open(p) as img:
            save_resized(img, os.path.join(train_dir, name), size)
        names['underwater'].append(name)
    return names


def build_ots(ots_zip, out, n_pairs, size, rng):
    """OTS: clear/<scene>.jpg + hazy/<scene>_<A>_<beta>.jpg (35 hazy versions per scene)."""
    haze_dir = os.path.join(out, 'train', 'haze')
    clear_dir = os.path.join(out, 'train', 'clear')
    os.makedirs(haze_dir, exist_ok=True)
    os.makedirs(clear_dir, exist_ok=True)

    with zipfile.ZipFile(ots_zip) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(IMG_EXTS)]
        clear = {os.path.splitext(os.path.basename(m))[0]: m
                 for m in members if m.split('/')[-2].lower() == 'clear'}
        hazy = sorted(m for m in members if m.split('/')[-2].lower() == 'hazy'
                      and os.path.basename(m).split('_')[0] in clear)
        if not clear or not hazy:
            raise RuntimeError(f"no clear/ or hazy/ folders found in {ots_zip}")
        print(f">>> OTS: {len(clear)} clear scenes, {len(hazy)} hazy images")

        picked = rng.sample(hazy, min(n_pairs, len(hazy)))
        names = []
        for m in tqdm(picked, desc='OTS pairs'):
            stem = os.path.splitext(os.path.basename(m))[0]
            name = f"{stem}.png"
            with zf.open(m) as f, Image.open(f) as img:
                save_resized(img, os.path.join(haze_dir, name), size)
            with zf.open(clear[stem.split('_')[0]]) as f, Image.open(f) as img:
                save_resized(img, os.path.join(clear_dir, name), size)
            names.append(name)
    return names


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--uieb_raw', required=True, help='folder with the UIEB raw-890 images')
    p.add_argument('--ruie_uiqs', required=True, help='RUIE UIQS folder (A..E subfolders)')
    p.add_argument('--exclude', nargs='*', default=[],
                   help='test-set folders (e.g. RUIE UCCS) whose images must not appear in training')
    p.add_argument('--ots_zip', required=True, help='Kaggle brunobelloni/outdoor-training-set-ots-reside zip')
    p.add_argument('--out', required=True)
    p.add_argument('--n_test', type=int, default=400, help='held-out underwater images (T400)')
    p.add_argument('--n_pairs', type=int, default=None,
                   help='OTS pairs; default = number of underwater training images')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    split = build_underwater([('UIEB', args.uieb_raw), ('RUIE', args.ruie_uiqs)],
                             args.exclude, args.out, args.n_test, args.size, rng)
    n_pairs = args.n_pairs or len(split['underwater'])
    split['ots'] = build_ots(args.ots_zip, args.out, n_pairs, args.size, rng)

    with open(os.path.join(args.out, 'split.json'), 'w') as f:
        json.dump({'seed': args.seed, **split}, f, indent=1)
    print(f">>> done: {len(split['underwater'])} underwater, {len(split['ots'])} OTS pairs, "
          f"{len(split['T400'])} T400 -> {args.out}")
