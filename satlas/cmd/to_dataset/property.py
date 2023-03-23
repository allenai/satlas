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
parser.add_argument("--ids", help="Limit outputs to this set of tiles and feature indexes", default=None)
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
args = parser.parse_args()

image_size = {
    'lowres': 512,
    'highres': 512,
}

image_types = common.get_image_types(args.satlas_root)
image_type_map = {image_meta['name']: image_meta for image_meta in image_types}

ids_ok = None
if args.ids:
    ids_ok = set()
    with open(args.ids, 'r') as f:
        for k, v in json.load(f).items():
            for tile_and_idx in v:
                ids_ok.add(k+'_'+tile_and_idx)

def process(job):
    satlas_tile, event_id, event_path, is_static = job

    vector_fname = os.path.join(event_path, 'vector.json')
    if not os.path.exists(vector_fname):
        return
    with open(vector_fname, 'r') as f:
        data = json.load(f)

    # Some dynamic labels don't have ImageName, but ImageName is required.
    if not is_static and 'ImageName' not in data['metadata']:
        return
    anchor_image_name = None
    if not is_static:
        anchor_image_name = data['metadata']['ImageName']

    for task in property_tasks:
        features = data.get(task['obj'], [])

        for feature_idx, feature in enumerate(features):
            if task['property'] not in feature.get('Properties', {}):
                continue

            # Check ids file but only for property tasks.
            # Since properties are limited to a subset for ones that have too many potential labels.
            if task['type'] != 'classify':
                image_id = '{}_{}_{}_{}_{}'.format(task['obj'], task['property'], satlas_tile[0], satlas_tile[1], feature_idx)
                if ids_ok is not None and image_id not in ids_ok:
                    continue

            value = feature['Properties'][task['property']]

            if task['type'] == 'classify':
                # Property for classification task is integer so let's convert it to the category name.
                value = task['categories'][int(value)]

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
                if task['image_type'] != 'all' and task['image_type'] != image_meta['name']:
                    continue

                factor = image_meta['factor']
                label_path = os.path.join(args.out_path, image_meta['name'], task['name'])

                feat_col = satlas_tile[0]*factor*512 + satlas_feat_col*factor//16
                feat_row = satlas_tile[1]*factor*512 + satlas_feat_row*factor//16
                sub_tile = (
                    feat_col//512,
                    feat_row//512,
                )

                if sub_tile not in image_meta['image_list']:
                    # Skip labels that definitely have no associated image.
                    # We may still produce excess labels since subsets of tiles are used for static / dynamic labels.
                    # But those splits will be specified when loading the dataset.
                    continue

                cur_out_dir = os.path.join(label_path, '{}_{}_{}_{}'.format(satlas_tile[0], satlas_tile[1], feature_idx, event_id))
                os.makedirs(cur_out_dir)
                with open(os.path.join(cur_out_dir, 'gt.txt'), 'w') as f:
                    f.write("{}\n".format(value))

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
                    cur_img_size = image_size[image_meta['name']]
                    for band in os.listdir(src_img_dir):
                        if band == 'tci':
                            num_channels = 3
                        else:
                            num_channels = 1

                        im = satlas.util.load_window(
                            base_dir=os.path.join(src_img_dir, band),
                            column=feat_col - cur_img_size//2,
                            row=feat_row - cur_img_size//2,
                            width=cur_img_size,
                            height=cur_img_size,
                            bands=num_channels,
                        )
                        skimage.io.imsave(
                            os.path.join(dst_img_dir, '{}.png'.format(band)),
                            im,
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
p = multiprocessing.Pool(args.workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
