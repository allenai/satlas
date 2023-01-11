import json
import math
import numpy as np
import scipy.optimize

from satlas.util import grid_index, geom

properties = [
    ['wind_turbine', 'rotor_diameter', 'numeric'],
    ['wind_turbine', 'power_mw', 'numeric'],
    ['parking_lot', 'capacity', 'numeric'],
    ['track', 'sport', 'category'],
    ['road', 'road_type', 'category'],
    ['road', 'lanes', 'numeric'],
    ['road', 'max_speed', 'numeric'],
    ['road', 'bridge', 'numeric'],
    ['power_plant', 'plant_type', 'category'],
    ['quarry', 'resource', 'category'],
    ['park', 'park_type', 'category'],
    ['park', 'sport', 'category'],
    ['smoke', 'smoke', 'classify'],
    ['snow', 'snow', 'classify']
]

def compare(job):
    gt_fname, pred_fname = job

    with open(gt_fname, 'r') as f:
        gt = json.load(f)
    with open(pred_fname, 'r') as f:
        pred = json.load(f)

    counts = {}

    for feat_name, prop_name, prop_type in properties:
        if feat_name not in gt:
            continue
        for gt_feature, pred_feature in zip(gt[feat_name], pred[feat_name]):
            if prop_name not in gt_feature.get('Properties', {}):
                continue

            k = '{}_{}_{}'.format(feat_name, prop_name, prop_type)
            if k not in counts:
                counts[k] = [0, 0]

            if prop_type == 'category' or prop_type == 'classify':
                if gt_feature['Properties'][prop_name] == pred_feature['Properties'].get(prop_name, 'invalid'):
                    correct = 1
                else:
                    correct = 0
                counts[k][0] += correct
                counts[k][1] += 1

            elif prop_type == 'numeric':
                gt_val = float(gt_feature['Properties'][prop_name])
                pred_val = float(pred_feature['Properties'][prop_name])
                epsilon = 0.01
                percent_error = abs(gt_val - pred_val) / max(gt_val, epsilon)
                accuracy = 1 - max(percent_error, 0)
                counts[k][0] += accuracy
                counts[k][1] += 1

            else:
                raise Exception('unknown property type {}'.format(prop_type))

    return counts

def get_scores(evaluator):
    if evaluator.lowres_only:
        # Properties can only be estimated with high-resolution imagery.
        return [], {}

    all_counts = evaluator.map(
        func=compare,
        fname='vector.json',
    )

    sums = {}
    for counts in all_counts:
        for label, (accuracy, total) in counts.items():
            if label not in sums:
                sums[label] = [0, 0]
            sums[label][0] += accuracy
            sums[label][1] += total

    label_scores = {}
    for label, (accuracy, total) in sums.items():
        avg_accuracy = accuracy / total
        label_scores[label] = avg_accuracy

    category_accuracy = np.mean([v for k, v in label_scores.items() if k.endswith('_category')])
    classify_accuracy = np.mean([v for k, v in label_scores.items() if k.endswith('_classify')])
    numeric_accuracy = np.mean([v for k, v in label_scores.items() if k.endswith('_numeric')])

    return [
        ('property_category_accuracy', category_accuracy),
        ('property_numeric_accuracy', numeric_accuracy),
        ('classify_accuracy', classify_accuracy),
    ], label_scores
