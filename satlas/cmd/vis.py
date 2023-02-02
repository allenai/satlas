import argparse
import json
import multiprocessing
import numpy as np
import os
import shutil
import skimage.draw
import skimage.io
import skimage.morphology
import tqdm

from satlas.tasks import detect_tasks, raster_tasks, property_tasks

tasks = {
    task['name']: task
    for task in detect_tasks + raster_tasks + property_tasks
}

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path to dataset made by to_dataset script.")
parser.add_argument("--task", help="Task name like polygon or vessel or dem.")
parser.add_argument("--out_path", help="Output directory.")
parser.add_argument("--workers", help="Number of worker processes.", type=int, default=1)
args = parser.parse_args()

task = tasks[args.task]
task_type = task['type']
stroke_width = 2

default_colors = [
    [0, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [128, 0, 128],
    [255, 255, 255],
    [0, 128, 0],
    [128, 128, 128],
    [165, 42, 42],
    [128, 0, 0],
    [255, 165, 0],
    [255, 105, 180],
    [192, 192, 192],
    [173, 216, 230],
    [32, 178, 170],
    [255, 0, 255],
    [128, 128, 0],
    [47, 79, 79],
    [255, 215, 0],
    [192, 192, 192],
    [240, 230, 140],
    [154, 205, 50],
    [64, 64, 64],
    [255, 165, 0],
    [0, 192, 0],
]

if 'colors' in task:
    colors = task['colors']
else:
    colors = default_colors

def process(image_id):
    # Find most recent image.
    image_fnames = []
    for fname in os.listdir(os.path.join(args.path, image_id)):
        if not fname.startswith('image_'):
            continue
        image_fnames.append(fname)
    image_fnames.sort()
    image_fname = os.path.join(args.path, image_id, image_fnames[-1])

    if task_type in ['property_numeric', 'property_category', 'classify']:
        with open(os.path.join(args.path, image_id, 'gt.txt'), 'r') as f:
            value = f.read().strip()
        shutil.copyfile(
            image_fname,
            os.path.join(args.out_path, '{}_im_{}.png'.format(image_id, value)),
        )
        return

    # Create gt image.
    if task_type in ['point', 'polygon']:
        gt = skimage.io.imread(image_fname)
        with open(os.path.join(args.path, image_id, 'gt.json'), 'r') as f:
            data = json.load(f)

        if task_type == 'point':
            for p in data:
                left, top, right, bottom, category = p[0:5]
                category_idx = task['categories'].index(category)
                color = colors[category_idx]

                left = np.clip(left, 0, gt.shape[1])
                top = np.clip(top, 0, gt.shape[0])
                right = np.clip(right, 0, gt.shape[1])
                bottom = np.clip(bottom, 0, gt.shape[0])
                gt[top:bottom, left:left+stroke_width, :] = color
                gt[top:bottom, right-stroke_width:right, :] = color
                gt[top:top+stroke_width, left:right, :] = color
                gt[bottom-stroke_width:bottom, left:right, :] = color

        elif task_type == 'polygon':
            mask = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)

            for p in data:
                coords = p[1]
                category = p[2]
                category_idx = task['categories'].index(category)
                color = colors[category_idx]

                exterior = np.array(coords[0], dtype=np.int32)
                rows, cols = skimage.draw.polygon(exterior[:, 1], exterior[:, 0], shape=(gt.shape[0], gt.shape[1]))
                mask[:, :] = 0
                mask[rows, cols] = 1

                for _ in range(stroke_width):
                    mask = skimage.morphology.binary_dilation(mask)
                gt[mask] = color

    elif task_type in ['segment', 'bin_segment', 'regress']:
        gt_raw = skimage.io.imread(os.path.join(args.path, image_id, 'gt.png'))

        if task_type in ['segment', 'bin_segment']:
            gt = np.zeros((gt_raw.shape[0], gt_raw.shape[1], 3), dtype=np.uint8)
            for category_idx in range(len(task['categories'])):
                color = colors[category_idx]
                if task_type == 'segment':
                    gt[gt_raw == category_idx] = color
                else:
                    mask_value = 2**category_idx
                    gt[gt_raw & mask_value] = color

        elif task_type == 'regress':
            gt = gt_raw

    shutil.copyfile(
        image_fname,
        os.path.join(args.out_path, '{}_im.png'.format(image_id)),
    )
    skimage.io.imsave(
        os.path.join(args.out_path, '{}_gt.png'.format(image_id)),
        gt,
        check_contrast=False,
    )

image_ids = os.listdir(args.path)
p = multiprocessing.Pool(args.workers)
for _ in tqdm.tqdm(p.imap_unordered(process, image_ids), total=len(image_ids)):
    pass
p.close()