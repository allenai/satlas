import json
import math
import numpy as np
import scipy.optimize

from satlas.util import grid_index, geom

lowres_categories = [
    "wind_turbine", "lighthouse", "mineshaft", "aerialway_pylon", "helipad",
    "communications_tower", "petroleum_well", "water_tower", "power_tower",
    'vessel',
]
all_categories = [
    'wind_turbine', 'lighthouse', 'mineshaft', 'aerialway_pylon', 'helipad',
    'fountain', 'toll_booth', 'chimney', 'communications_tower',
    'flagpole', 'petroleum_well', 'water_tower', 'street_lamp',
    'traffic_signals', 'power_tower',
    'airplane', 'rooftop_solar_panel', 'vessel',
]

def compare_points(gt, pred):
    def to_points(l):
        points = []
        for feature in l:
            if feature['Geometry']['Type'] != 'point':
                continue
            points.append(feature['Geometry']['Point'])
        return points

    gt_points = to_points(gt)
    pred_points = to_points(pred)
    distance_threshold = 128

    pred_index = grid_index.GridIndex(size=256)
    for i, (col, row) in enumerate(pred_points):
        pred_index.insert(geom.Point(col, row), i)

    match_matrix = np.ones((len(gt_points), len(pred_points)), dtype=np.float32)
    for i, (gt_col, gt_row) in enumerate(gt_points):
        rect = geom.Point(gt_col, gt_row).bounds().add_tol(distance_threshold)
        options = pred_index.search(rect)
        for j in options:
            pred_col, pred_row = pred_points[j]
            distance = math.sqrt((pred_col - gt_col)**2 + (pred_row - gt_row)**2)
            if distance > distance_threshold:
                continue
            match_matrix[i, j] = 0

    rows, cols = scipy.optimize.linear_sum_assignment(match_matrix)

    num_tp = len([ii for ii in range(len(rows)) if match_matrix[rows[ii], cols[ii]] == 0])
    num_fp = len(pred_points) - num_tp
    num_fn = len(gt_points) - num_tp

    return num_tp, num_fp, num_fn

def compare(job):
    gt_fname, pred_fname, categories = job

    with open(gt_fname, 'r') as f:
        gt = json.load(f)

    with open(pred_fname, 'r') as f:
        pred = json.load(f)

    counts = {}
    for category in categories:
        tp, fp, fn = compare_points(
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

    return [('point_f1', overall_f1)], category_scores
