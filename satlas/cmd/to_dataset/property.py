import argparse
import datetime
import json
import multiprocessing
import numpy as np
import os
import random
import skimage.io
import tqdm

import satlas.util
from satlas.tasks import property_tasks
from satlas.cmd.to_dataset import common

parser = argparse.ArgumentParser()
parser.add_argument("--satlas_root", help="Satlas root directory.")
parser.add_argument("--split", help="Satlas split.")
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
args = parser.parse_args()

image_size = 512

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

        for task in property_tasks:
            features = data.get(task['obj'], [])

            for feature_idx, feature in enumerate(features):
                if task['property'] not in feature.get('Properties', {}):
                    continue

                image_id = '{}_{}_{}'.format(satlas_tile[0], satlas_tile[1], feature_idx)
                value = feature['Properties'][task['property']]

                # Get center of the feature.
                points = []
                geometry = feature['Geometry']
                if geometry['Type'] == 'point':
                    points.append(geometry['Point'])
                if geometry['Type'] == 'polygon':
                    for ring in geometry['Polygon']:
                        points.extend(ring)
                if geometry['Type'] == 'polyline':
                    # For polyline just use the middle point.
                    # Since we want the image to be centered at least on some point along the polyline and not somewhere else in case it is spiraly shape.
                    polyline = geometry['Polyline']
                    points.append(polyline[len(polyline)//2])
                sx = min([p[0] for p in points])
                sy = min([p[1] for p in points])
                ex = max([p[0] for p in points])
                ey = max([p[1] for p in points])
                satlas_feat_col = (sx + ex) // 2
                satlas_feat_row = (sy + ey) // 2

                for image_meta in image_types:
                    factor = image_meta['factor']
                    image_path = os.path.join(args.satlas_root, args.split, image_meta['name'])
                    label_path = os.path.join(args.out_path, '{}_{}'.format(image_meta['name'], task['name']))

                    feat_col = satlas_tile[0]*factor*512 + satlas_feat_col*factor//16
                    feat_row = satlas_tile[1]*factor*512 + satlas_feat_row*factor//16
                    sub_tile = (
                        feat_col//512,
                        feat_row//512,
                    )

                    image_uuids = common.get_image_uuids(image_meta, sub_tile, ts1, ts2)
                    if not image_uuids:
                        continue

                    cur_out_dir = os.path.join(label_path, image_id)
                    os.makedirs(cur_out_dir, exist_ok=True)
                    with open(os.path.join(cur_out_dir, 'gt.txt'), 'w') as f:
                        f.write("{}\n".format(value))

                    for image_uuid in image_uuids:
                        image_time = image_meta['image_times'][image_uuid]
                        dst_fname = os.path.join(cur_out_dir, 'image_{}.png'.format(image_time))
                        if os.path.exists(dst_fname):
                            continue
                        im = satlas.util.load_window(
                            base_dir=os.path.join(image_path, image_uuid, 'tci'),
                            column=feat_col - image_size//2,
                            row=feat_row - image_size//2,
                            width=image_size,
                            height=image_size,
                        )
                        skimage.io.imsave(dst_fname, im)

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
