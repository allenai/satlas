import json
import math
import numpy as np
import skimage.draw
import skimage.morphology

from satlas.util import grid_index, geom

lowres_categories = []
all_categories = [
    'airport_runway', 'airport_taxiway', 'raceway', 'road', 'railway', 'river',
]

def compare_polylines(gt, pred):
    def to_mask(l):
        mask = np.zeros((2048, 2048), dtype=bool)
        for feature in l:
            if feature['Geometry']['Type'] != 'polyline':
                continue
            if len(feature['Geometry']['Polyline']) < 2:
                continue
            polyline = [(int(x//4), int(y//4)) for x, y in feature['Geometry']['Polyline']]
            for i in range(len(polyline)-1):
                rows, cols = skimage.draw.line(polyline[i][1], polyline[i][0], polyline[i+1][1], polyline[i+1][0])
                valid_indices = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
                mask[rows[valid_indices], cols[valid_indices]] = True
        for _ in range(4):
            mask = skimage.morphology.binary_dilation(mask)
        return mask

    gt_mask = to_mask(gt)
    pred_mask = to_mask(pred)

    tp = np.count_nonzero(gt_mask & pred_mask)
    fp = np.count_nonzero((~gt_mask) & pred_mask)
    fn = np.count_nonzero(gt_mask & (~pred_mask))
    return tp, fp, fn

def compare(job):
    gt_fname, pred_fname, categories = job

    with open(gt_fname, 'r') as f:
        gt = json.load(f)

    with open(pred_fname, 'r') as f:
        pred = json.load(f)

    counts = {}
    for category in categories:
        tp, fp, fn = compare_polylines(
            gt.get(category, []),
            pred.get(category, []),
        )
        counts[category] = (tp, fp, fn)

    return counts

def get_scores(evaluator):
    if evaluator.lowres_only:
        categories = lowres_categories
    else:
        categories = all_categories

    all_counts = evaluator.map(
        func=compare,
        fname='vector.json',
        args=[categories],
    )

    sums = {}
    for counts in all_counts:
        for category, (tp, fp, fn) in counts.items():
            if category not in sums:
                sums[category] = [0, 0, 0]
            sums[category][0] += tp
            sums[category][1] += fp
            sums[category][2] += fn

    category_scores = {}
    for category, (tp, fp, fn) in sums.items():
        # Skip categories with not enough ground truth points in the current split.
        gt_total = tp + fn
        if gt_total < 20:
            continue

        # Compute precision and recall, and F1 score.
        if tp == 0:
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        category_scores[category] = f1

    overall_f1 = np.mean(list(category_scores.values()))

    return [('polyline_f1', overall_f1)], category_scores
