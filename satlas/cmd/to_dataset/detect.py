import argparse
import datetime
import json
import multiprocessing
import numpy as np
import os
import random
import tqdm

from satlas.tasks import detect_tasks, polyline_tasks
from satlas.cmd.to_dataset import common

parser = argparse.ArgumentParser()
parser.add_argument("--satlas_root", help="Satlas root directory.")
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
args = parser.parse_args()

box_size = 20

category_to_task = {}
for task in (detect_tasks + polyline_tasks):
    for category in task['categories']:
        category_to_task[category] = task
task_map = {task['name']: task for task in (detect_tasks + polyline_tasks)}

image_types = common.get_image_types(args.satlas_root)
image_type_map = {image_meta['name']: image_meta for image_meta in image_types}

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

    # Map from (tile, image_type, taskname) -> label list.
    labels_grouped = {}

    for category, labels in data.items():
        if category not in category_to_task:
            continue

        task = category_to_task[category]
        task_name = task['name']
        modality = task['type']
        task_image_type = task['image_type']

        for image_meta in image_types:
            if task_image_type != 'all' and task_image_type != image_meta['name']:
                continue

            factor = image_meta['factor']

            for label in labels:
                if label['Geometry']['Type'] != modality:
                    continue

                if modality == 'polygon' or modality == 'polyline':
                    if modality == 'polygon':
                        coords = label['Geometry']['Polygon']
                        coords = [[(int(col), int(row)) for col, row in ring] for ring in coords]
                        exterior = coords[0]
                    elif modality == 'polyline':
                        coords = label['Geometry']['Polyline']
                        coords = [(int(col), int(row)) for col, row in coords]
                        exterior = coords

                    # Create a unique ID for this feature, even across tiles.
                    # This is because when loading windows greater than the size of one tile,
                    # we may end up loading the same polygon/polyline from different tiles.
                    # The unique ID can be used to eliminate those duplicate polygons/polylines.
                    # To get unique ID, we just append the absolute coordinates of first point with category.
                    absolute_coords = (
                        satlas_tile[0]*8192 + exterior[0][0],
                        satlas_tile[1]*8192 + exterior[0][1],
                    )
                    feat_id = '{}_{}_{}'.format(absolute_coords[0], absolute_coords[1], category)

                    tile_bounds = (
                        np.clip(min([p[0] for p in exterior])*factor//8192, 0, factor-1),
                        np.clip(min([p[1] for p in exterior])*factor//8192, 0, factor-1),
                        np.clip(max([p[0] for p in exterior])*factor//8192+1, 0, factor),
                        np.clip(max([p[1] for p in exterior])*factor//8192+1, 0, factor),
                    )

                    for off_col in range(tile_bounds[0], tile_bounds[2]):
                        for off_row in range(tile_bounds[1], tile_bounds[3]):
                            sub_tile = (
                                satlas_tile[0]*factor + off_col,
                                satlas_tile[1]*factor + off_row,
                            )

                            k = (sub_tile, image_meta['name'], task_name)
                            if k not in labels_grouped:
                                labels_grouped[k] = []

                            if modality == 'polygon':
                                out_coords = [[(x*factor//16 - off_col*512, y*factor//16 - off_row*512) for x, y in ring] for ring in coords]
                            elif modality == 'polyline':
                                out_coords = [(x*factor//16 - off_col*512, y*factor//16 - off_row*512) for x, y in coords]

                            labels_grouped[k].append([
                                feat_id,
                                out_coords,
                                category,
                                label.get('Properties', {}),
                            ])

                elif modality == 'point':
                    x, y = label['Geometry']['Point']
                    tile_offset = (
                        x*factor//8192,
                        y*factor//8192,
                    )
                    if tile_offset[0] < 0 or tile_offset[0] >= factor or tile_offset[1] < 0 or tile_offset[1] >= factor:
                        continue

                    sub_tile = (
                        satlas_tile[0]*factor + tile_offset[0],
                        satlas_tile[1]*factor + tile_offset[1],
                    )

                    k = (sub_tile, image_meta['name'], task_name)
                    if k not in labels_grouped:
                        labels_grouped[k] = []
                    center = (
                        x*factor//16 - tile_offset[0]*512,
                        y*factor//16 - tile_offset[1]*512,
                    )
                    labels_grouped[k].append([
                        center[0] - box_size,
                        center[1] - box_size,
                        center[0] + box_size,
                        center[1] + box_size,
                        category,
                        label.get('Properties', {}),
                    ])

    for (tile, image_type, task_name), labels in labels_grouped.items():
        image_meta = image_type_map[image_type]
        label_path = os.path.join(args.out_path, image_type, task_name)

        if tile not in image_meta['image_list']:
            # Skip labels that definitely have no associated image.
            # We may still produce excess labels since subsets of tiles are used for static / dynamic labels.
            # But those splits will be specified when loading the dataset.
            continue

        cur_out_dir = os.path.join(label_path, '{}_{}_{}'.format(event_id, tile[0], tile[1]))
        os.makedirs(cur_out_dir)
        with open(os.path.join(cur_out_dir, 'gt.json'), 'w') as f:
            json.dump(labels, f)

        for image_name in image_meta['image_list'][tile]:
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
                src_img_fname = os.path.join(src_img_dir, band, '{}_{}.png'.format(tile[0], tile[1]))
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
