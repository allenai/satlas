import argparse
import datetime
import json
import multiprocessing
import numpy as np
import os
import tqdm

from satlas.tasks import detect_tasks
from satlas.cmd.to_dataset import common

parser = argparse.ArgumentParser()
parser.add_argument("--satlas_root", help="Satlas root directory.")
parser.add_argument("--split", help="Satlas split.")
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
parser.add_argument("--box_size", help="Bounding box size for points.", type=int, default=20)
args = parser.parse_args()

category_to_task = {}
for task in detect_tasks:
    for category in task['categories']:
        category_to_task[category] = task
task_map = {task['name']: task for task in detect_tasks}

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
                            exterior = coords[0]
                        elif modality == 'polyline':
                            coords = label['Geometry']['Polyline']
                            exterior = coords

                        # Create a unique ID for this feature, even across tiles.
                        # This is because when span>1 we may load the same polygon/polyline from different tiles.
                        # And we want to be able to remove those duplicates in multisat dataset code.
                        # To get unique ID, we just append the absolute coordinates of first point with category.
                        absolute_coords = (
                            satlas_tile[0]*8192 + exterior[0][0],
                            satlas_tile[1]*8192 + exterior[0][1],
                        )
                        feat_id = '{}_{}_{}'.format(absolute_coords[0], absolute_coords[1], category)

                        tile_bounds = (
                            np.clip(min([p[0] for p in exterior])*factor//8192, 0, factor),
                            np.clip(min([p[1] for p in exterior])*factor//8192, 0, factor),
                            np.clip(max([p[0] for p in exterior])*factor//8192+1, 0, factor+1),
                            np.clip(max([p[1] for p in exterior])*factor//8192+1, 0, factor+1),
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
                            center[0] - args.box_size,
                            center[1] - args.box_size,
                            center[0] + args.box_size,
                            center[1] + args.box_size,
                            category,
                            label.get('Properties', {}),
                        ])

        for (tile, image_type, task_name), labels in labels_grouped.items():
            image_meta = image_type_map[image_type]
            label_path = os.path.join(args.out_path, '{}_{}'.format(image_type, task_name))

            image_uuids = common.get_image_uuids(image_meta, tile, ts1, ts2)
            if not image_uuids:
                continue

            cur_out_dir = os.path.join(label_path, '{}_{}_{}'.format(tile[0], tile[1], event_id))
            os.makedirs(cur_out_dir, exist_ok=True)
            with open(os.path.join(cur_out_dir, 'gt.json'), 'w') as f:
                json.dump(labels, f)

            for image_uuid in image_uuids:
                image_time = image_meta['image_times'][image_uuid]
                dst_fname = os.path.join(cur_out_dir, 'image_{}.png'.format(image_time))
                if os.path.exists(dst_fname):
                    continue
                os.symlink(
                    os.path.join(args.satlas_root, args.split, image_type, image_uuid, 'tci', '{}_{}.png'.format(tile[0], tile[1])),
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
