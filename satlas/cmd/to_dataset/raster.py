import argparse
import datetime
import json
import multiprocessing
import numpy as np
import os
import skimage.io
import tqdm

from satlas.tasks import raster_tasks
from satlas.cmd.to_dataset import common

parser = argparse.ArgumentParser()
parser.add_argument("--satlas_root", help="Satlas root directory.")
parser.add_argument("--split", help="Satlas split.")
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
args = parser.parse_args()

image_types = common.get_image_types(args.satlas_root, args.split)
image_type_map = {image_meta['name']: image_meta for image_meta in image_types}

def process(job):
    satlas_tile, tile_path = job

    for event_id in os.listdir(tile_path):
        event_path = os.path.join(tile_path, event_id)

        vector_fname = os.path.join(event_path, 'vector.json')
        with open(vector_fname, 'r') as f:
            data = json.load(f)

        ts1 = datetime.datetime.fromisoformat(data['metadata']['Start']).replace(tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime.fromisoformat(data['metadata']['End']).replace(tzinfo=datetime.timezone.utc)

        for task in raster_tasks:
            task_name = task['name']
            task_image_type = task['image_type']

            src_fname = os.path.join(event_path, task['id']+'.png')
            if not os.path.exists(src_fname):
                continue
            im = skimage.io.imread(src_fname)

            for image_meta in image_types:
                if task_image_type != 'all' and task_image_type != image_meta['name']:
                    continue

                factor = image_meta['factor']
                label_path = os.path.join(args.out_path, '{}_{}'.format(image_meta['name'], task_name))

                for off_col in range(factor):
                    for off_row in range(factor):
                        sub_tile = (
                            satlas_tile[0]*factor + off_col,
                            satlas_tile[1]*factor + off_row,
                        )

                        image_uuids = common.get_image_uuids(image_meta, sub_tile, ts1, ts2)
                        if not image_uuids:
                            continue

                        crop = im[off_row*(512//factor):(off_row+1)*(512//factor), off_col*(512//factor):(off_col+1)*(512//factor)]
                        if crop.max() == 0:
                            # The whole label is invalid, so let's skip this output.
                            continue
                        crop = crop.repeat(repeats=factor, axis=0).repeat(repeats=factor, axis=1)

                        cur_out_dir = os.path.join(label_path, os.path.join('{}_{}_{}'.format(sub_tile[0], sub_tile[1], event_id)))
                        os.makedirs(cur_out_dir, exist_ok=True)
                        dst_fname = os.path.join(cur_out_dir, 'gt.png')
                        skimage.io.imsave(dst_fname, crop, check_contrast=False)

                        for image_uuid in image_uuids:
                            image_time = image_meta['image_times'][image_uuid]
                            dst_fname = os.path.join(cur_out_dir, 'image_{}.png'.format(image_time))
                            if os.path.exists(dst_fname):
                                continue
                            os.symlink(
                                os.path.join(args.satlas_root, args.split, image_meta['name'], image_uuid, 'tci', '{}_{}.png'.format(sub_tile[0], sub_tile[1])),
                                dst_fname,
                            )

print('populate jobs')
jobs = []
in_path = os.path.join(args.satlas_root, '{}_labels'.format(args.split))
for tile_str in os.listdir(in_path):
    parts = tile_str.split('_')
    satlas_tile = (int(parts[0]), int(parts[1]))
    tile_path = os.path.join(in_path, tile_str)
    jobs.append((satlas_tile, tile_path))

print('process')
p = multiprocessing.Pool(args.workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
