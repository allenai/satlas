import json
import math
import numpy as np
import scipy.optimize
import shapely.geometry

from satlas.util import grid_index, geom

lowres_categories = [
    "aquafarm", "lock", "dam", "solar_farm", "power_plant", "park",
    "parking_garage", "parking_lot", "landfill", "quarry", "stadium",
    "airport", "airport_apron", "airport_terminal", "ski_resort", "theme_park",
    "track", "wastewater_plant",
]
all_categories = [
    'aquafarm', 'lock', 'dam', 'solar_farm', 'power_plant', 'gas_station',
    'park', 'parking_garage', 'parking_lot', 'landfill', 'quarry', 'stadium',
    'airport', 'airport_apron', 'airport_hangar', 'airport_terminal',
    'ski_resort', 'theme_park', 'storage_tank', 'silo', 'track',
    'wastewater_plant', 'power_substation', 'building',
    'pier',
]

def get_iou(shp1, shp2):
    if not shp1.intersects(shp2):
        return 0
    return shp1.intersection(shp2).area / shp1.union(shp2).area

def compare_polygons(gt, pred):
    pixels_per_tile = 512 * 16
    tile_shp = shapely.geometry.box(0, 0, pixels_per_tile, pixels_per_tile)

    def to_shapes(l):
        shps = []
        for feature in l:
            if feature['Geometry']['Type'] != 'polygon':
                continue
            if len(feature['Geometry']['Polygon'][0]) < 3:
                continue
            exterior = feature['Geometry']['Polygon'][0]
            interiors = feature['Geometry']['Polygon'][1:]
            shp = shapely.geometry.Polygon(shell=exterior, holes=interiors).buffer(1.0)
            shp = shp.intersection(tile_shp)
            if shp.area <= 0:
                continue
            shps.append(shp.buffer(1.0))
        return shps

    gt_shps = to_shapes(gt)
    pred_shps = to_shapes(pred)

    pred_index = grid_index.GridIndex(size=256)
    for i, shp in enumerate(pred_shps):
        rect = geom.Rectangle(
            geom.Point(shp.bounds[0], shp.bounds[1]),
            geom.Point(shp.bounds[2], shp.bounds[3]),
        )
        pred_index.insert_rect(rect, i)

    # Mark matches 0 and mismatches 1.
    # Then use linear_sum_assignment to match gt to pred.
    # We also compute the IoUs between gt polygon and nearest predicted polygons.
    # Average IoU will be mean of these IoUs + 0s for any unmatched gt and predicted polygons.
    match_matrix = np.ones((len(gt_shps), len(pred_shps)), dtype=np.float32)
    iou_scores = []
    for i, gt_shp in enumerate(gt_shps):
        rect = geom.Rectangle(
            geom.Point(gt_shp.bounds[0], gt_shp.bounds[1]),
            geom.Point(gt_shp.bounds[2], gt_shp.bounds[3]),
        )
        options = pred_index.search(rect)
        best_iou = 0
        for j in options:
            pred_shp = pred_shps[j]
            iou = get_iou(gt_shp, pred_shp)
            if iou < 0.1:
                continue
            match_matrix[i, j] = 0

            if iou > best_iou:
                best_iou = iou

        iou_scores.append(best_iou)

    rows, cols = scipy.optimize.linear_sum_assignment(match_matrix)

    num_tp = len([ii for ii in range(len(rows)) if match_matrix[rows[ii], cols[ii]] == 0])
    num_fp = len(pred_shps) - num_tp
    num_fn = len(gt_shps) - num_tp

    for _ in range(num_fp):
        iou_scores.append(0)
    if len(iou_scores) == 0:
        avg_iou = None
    else:
        avg_iou = np.mean(iou_scores)

    return num_tp, num_fp, num_fn, avg_iou

def compare(job):
    gt_fname, pred_fname, categories = job

    with open(gt_fname, 'r') as f:
        gt = json.load(f)

    with open(pred_fname, 'r') as f:
        pred = json.load(f)

    counts = {}
    for category in categories:
        tp, fp, fn, iou = compare_polygons(
            gt.get(category, []),
            pred.get(category, []),
        )
        counts[category] = (tp, fp, fn, iou)

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
        for category, (tp, fp, fn, iou) in counts.items():
            if category not in sums:
                sums[category] = [0, 0, 0, 0, 0]
            sums[category][0] += tp
            sums[category][1] += fp
            sums[category][2] += fn

            if iou is not None:
                sums[category][3] += iou
                sums[category][4] += 1

    category_scores = {}
    for category, (tp, fp, fn, iou, num_tiles) in sums.items():
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

        avg_iou = iou / num_tiles

        category_scores[category+'_f1'] = f1
        category_scores[category+'_iou'] = avg_iou

    overall_f1 = np.mean([v for k, v in category_scores.items() if k.endswith('_f1')])
    overall_iou = np.mean([v for k, v in category_scores.items() if k.endswith('_iou')])

    return [
        ('polygon_f1', overall_f1),
        ('polygon_iou', overall_iou),
    ], category_scores
