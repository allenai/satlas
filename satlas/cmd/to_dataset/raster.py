import argparse
import datetime
import json
import multiprocessing
import numpy as np
import os
import skimage.io
import random
import tqdm

from satlas.tasks import raster_tasks
from satlas.cmd.to_dataset import common

parser = argparse.ArgumentParser()
parser.add_argument("--satlas_root", help="Satlas root directory.")
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
args = parser.parse_args()

image_types = common.get_image_types(args.satlas_root)
image_type_map = {image_meta['name']: image_meta for image_meta in image_types}

def process(job):
    satlas_tile, event_id, event_path, is_static = job

    anchor_image_name = None
    if not is_static:
        vector_fname = os.path.join(event_path, 'vector.json')
        with open(vector_fname, 'r') as f:
            data = json.load(f)

        # Some dynamic labels don't have ImageName, but ImageName is required.
        if 'ImageName' not in data['metadata']:
            return

        anchor_image_name = data['metadata']['ImageName']

    for task in raster_tasks:
        task_name = task['name']
        task_image_type = task['image_type']

        if 'id' not in task:
            continue

        src_fname = os.path.join(event_path, task['id']+'.png')
        if not os.path.exists(src_fname):
            continue
        im = skimage.io.imread(src_fname)

        for image_meta in image_types:
            if task_image_type != 'all' and task_image_type != image_meta['name']:
                continue

            factor = image_meta['factor']
            label_path = os.path.join(args.out_path, image_meta['name'], task_name)

            for off_col in range(factor):
                for off_row in range(factor):
                    sub_tile = (
                        satlas_tile[0]*factor + off_col,
                        satlas_tile[1]*factor + off_row,
                    )

                    if sub_tile not in image_meta['image_list']:
                        # Skip labels that definitely have no associated image.
                        # We may still produce excess labels since subsets of tiles are used for static / dynamic labels.
                        # But those splits will be specified when loading the dataset.
                        continue

                    crop = im[off_row*(512//factor):(off_row+1)*(512//factor), off_col*(512//factor):(off_col+1)*(512//factor)]
                    if crop.max() == 0:
                        # The whole label is invalid, so let's skip this output.
                        continue
                    crop = crop.repeat(repeats=factor, axis=0).repeat(repeats=factor, axis=1)

                    cur_out_dir = os.path.join(label_path, '{}_{}_{}'.format(event_id, sub_tile[0], sub_tile[1]))
                    os.makedirs(cur_out_dir)
                    dst_fname = os.path.join(cur_out_dir, 'gt.png')
                    skimage.io.imsave(dst_fname, crop, check_contrast=False)

                    for image_name in image_meta['image_list'][sub_tile]:
                        prefix = 'image'
                        if anchor_image_name == image_name:
                            prefix = 'anchor'
                        ts_str = image_meta['image_times'][image_name].isoformat()[0:10]
                        dst_img_dir = os.path.join(cur_out_dir, 'images', '{}_{}'.format(prefix, ts_str))

                        if os.path.exists(dst_img_dir):
                            # Probably there are two images with the same timestamp.
                            # So it is okay to skip here.
                            continue

                        os.makedirs(dst_img_dir)

                        src_img_dir = os.path.join(args.satlas_root, image_meta['label'], image_name)
                        for band in os.listdir(src_img_dir):
                            src_img_fname = os.path.join(src_img_dir, band, '{}_{}.png'.format(sub_tile[0], sub_tile[1]))
                            if not os.path.exists(src_img_fname):
                                continue
                            os.symlink(
                                os.path.abspath(src_img_fname),
                                os.path.join(dst_img_dir, '{}.png'.format(band)),
                            )

jobs = []
print('populate static jobs')
static_path = os.path.join(args.satlas_root, 'static')
for tile_str in os.listdir(static_path):
    parts = tile_str.split('_')
    satlas_tile = (int(parts[0]), int(parts[1]))
    tile_path = os.path.join(static_path, tile_str)
    jobs.append((satlas_tile, tile_str, tile_path, True))
print('populate dynamic jobs')
dynamic_path = os.path.join(args.satlas_root, 'dynamic')
for tile_str in os.listdir(dynamic_path):
    parts = tile_str.split('_')
    satlas_tile = (int(parts[0]), int(parts[1]))
    for event_id in os.listdir(os.path.join(dynamic_path, tile_str)):
        event_path = os.path.join(dynamic_path, tile_str, event_id)
        jobs.append((satlas_tile, tile_str+'_'+event_id, event_path, False))

print('process')
random.shuffle(jobs)
p = multiprocessing.Pool(args.workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
