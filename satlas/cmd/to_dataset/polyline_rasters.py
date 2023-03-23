import argparse
import json
import multiprocessing
import numpy as np
import os
import skimage.draw
import skimage.io
import skimage.morphology
import tqdm

import satlas.util

tile_size = 512
highres_padding = 3
lowres_padding = 2
categories = [
    'road',
    'railway',
    'river',
    'airport_runway',
    'airport_taxiway',
    'raceway',
]

def handle_example(job):
    in_dir, out_dir, padding = job

    os.makedirs(out_dir, exist_ok=True)
    os.symlink(
        os.path.abspath(os.path.join(in_dir, 'images')),
        os.path.join(out_dir, 'images'),
    )

    im = np.zeros((tile_size, tile_size, len(categories)), dtype=bool)

    with open(os.path.join(in_dir, 'gt.json')) as f:
        for p in json.load(f):
            coords = p[1]
            category = p[2]

            category_idx = categories.index(category)
            for i in range(len(coords)-1):
                rows, cols = skimage.draw.line(coords[i][1], coords[i][0], coords[i+1][1], coords[i+1][0])
                valid = (rows >= 0) & (rows < tile_size) & (cols >= 0) & (cols < tile_size)
                im[rows[valid], cols[valid], category_idx] = True

    # Apply padding and combine into one image.
    for category_idx in range(im.shape[2]):
        for _ in range(padding):
            im[:, :, category_idx] = skimage.morphology.binary_dilation(im[:, :, category_idx])

    # Convert to encoded bitwise format.
    out_im = satlas.util.encode_multiclass_binary(im).astype(np.uint8)

    skimage.io.imsave(
        os.path.join(out_dir, 'gt.png'),
        out_im,
        check_contrast=False,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", help="Path to datasets like satlas_root/datasets/.")
    parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
    args = parser.parse_args()

    jobs = []
    for image_mode, padding in [('lowres', lowres_padding), ('highres', highres_padding)]:
        src_dir = os.path.join(args.dataset_root, image_mode, 'polyline')
        dst_dir = os.path.join(args.dataset_root, image_mode, 'polyline_bin_segment')
        for example_id in os.listdir(src_dir):
            jobs.append((
                os.path.join(src_dir, example_id),
                os.path.join(dst_dir, example_id),
                padding,
            ))

    p = multiprocessing.Pool(args.workers)
    out = p.imap_unordered(handle_example, jobs)
    for _ in tqdm.tqdm(out, total=len(jobs)):
        pass
    p.close()