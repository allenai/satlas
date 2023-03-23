import argparse
import json
import multiprocessing
import numpy as np
import os
import skimage.io
import sys
import tqdm

import satlas.model.dataset
import satlas.util

# Compute weight to balance classes across all tasks.
max_weight = 20

def get_classes(job):
    example_name, example_dir, task_type, num_categories = job
    cur = []

    if task_type == 'bin_segment':
        raw_im = skimage.io.imread(os.path.join(example_dir, 'gt.png'))
        im = satlas.util.decode_multiclass_binary(raw_im, num_categories)
        for cls_id in range(num_categories):
            if np.count_nonzero(im[:, :, cls_id]) == 0:
                continue
            cur.append('{}_{}'.format(task_name, cls_id))

    elif task_type == 'segment':
        im = skimage.io.imread(os.path.join(example_dir, 'gt.png'))
        for cls_id in range(num_categories):
            if np.count_nonzero(im == cls_id) == 0:
                continue
            cur.append('{}_{}'.format(task_name, cls_id))

    elif task_type == 'instance':
        with open(os.path.join(example_dir, 'gt.json'), 'r') as f:
            polygons = json.load(f)
        seen = set()
        for (polygon_id, coords, class_label, properties) in polygons:
            if class_label in seen:
                continue
            cur.append('{}_{}'.format(task_name, class_label))
            seen.add(class_label)

    elif task_type == 'detect':
        with open(os.path.join(example_dir, 'gt.json'), 'r') as f:
            boxes = json.load(f)
        seen = set()
        for box in boxes:
            if box[4] in seen:
                continue
            cur.append('{}_{}'.format(task_name, box[4]))
            seen.add(box[4])

    else:
        raise Exception('unknown task type ' + task_type)

    return (example_name, cur)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute tile weights to balance examples by category frequency.")
    parser.add_argument("--dataset_path", help="Path to datasets like /satlas/datasets/lowres/")
    parser.add_argument("--out_path", help="Output filename like bal_weights/lowres.json")
    args = parser.parse_args()

    tile_classes = {}
    class_counts = {}

    def add_class(example_name, cls_name):
        if example_name not in tile_classes:
            tile_classes[example_name] = []

        if cls_name is not None:
            tile_classes[example_name].append(cls_name)

            if cls_name not in class_counts:
                class_counts[cls_name] = 0
            class_counts[cls_name] += 1

    for task_name in os.listdir(args.dataset_path):
        if task_name not in satlas.model.dataset.tasks:
            continue

        task_dir = os.path.join(args.dataset_path, task_name)
        print(task_dir)
        task = satlas.model.dataset.tasks[task_name]

        if task['type'] in ['regress', 'classification']:
            for example_id in os.listdir(task_dir):
                example_name = '{}_{}'.format(task_name, example_id)
                add_class(example_name, task_name)

        else:
            jobs = []
            for example_id in os.listdir(task_dir):
                example_name = '{}_{}'.format(task_name, example_id)

                if 'categories' in task:
                    num_categories = len(task['categories'])
                else:
                    num_categories = None

                jobs.append((example_name, os.path.join(task_dir, example_id), task['type'], num_categories))

                # Make sure this tile is added here so it'll still appear in the class
                # map (but with weight zero) in case it has no classes.
                add_class(example_name, None)

            p = multiprocessing.Pool(64)
            for example_name, cls_list in tqdm.tqdm(p.imap(get_classes, jobs), total=len(jobs)):
                for cls_name in cls_list:
                    add_class(example_name, cls_name)
            p.close()

    # Compute class weights based on counts.
    # We will then set weight of tile to the maximum class weight.
    peak_cls = max(class_counts.values())
    class_weights = {
        cls_name: min(peak_cls / count, max_weight)
        for cls_name, count in class_counts.items()
    }

    tile_weights = {}
    for tile_str, classes in tile_classes.items():
        weight = 0
        for cls_id in classes:
            weight = max(weight, class_weights[cls_id])
        tile_weights[tile_str] = weight

    with open(args.out_path, 'w') as f:
        json.dump(tile_weights, f)
