import numpy as np
import os
import skimage.io

from satlas.tasks import raster_tasks
from satlas.util import grid_index, geom

def compare(job):
    gt_path, pred_path = job
    counts = {}

    for task in raster_tasks:
        task_name = task['name']
        task_id = task['id']
        task_type = task['type']

        gt_fname = os.path.join(gt_path, task_id+'.png')
        pred_fname = os.path.join(pred_path, task_id+'.png')

        if not os.path.exists(gt_fname):
            continue

        gt_im = skimage.io.imread(gt_fname)

        if os.path.exists(pred_fname):
            pred_im = skimage.io.imread(pred_fname)
        else:
            print('warning: missing prediction corresponding to {}'.format(gt_fname))
            pred_im = np.zeros((512, 512), dtype=np.uint8)

        if task_type == 'regress':
            mask = gt_im > 0
            error_im = np.abs(gt_im.astype(np.int32) - pred_im.astype(np.int32))
            error = (error_im * mask.astype(np.int32)).sum() / np.count_nonzero(mask)
            counts[task_name+'_error'] = (error, 1, 1)

        elif task_type == 'segment':
            mask = gt_im > 0
            for cls_id, cls_name in enumerate(task['categories']):
                if cls_name == 'background' or cls_name == 'invalid':
                    continue

                gt_bin = gt_im == cls_id
                pred_bin = pred_im == cls_id
                tp_im = (gt_bin) & (pred_bin) & mask
                fp_im = (~gt_bin) & (pred_bin) & mask
                fn_im = (gt_bin) & (~pred_bin) & mask
                counts['{}_{}_f1'.format(task_name, cls_name)] = (np.count_nonzero(tp_im), np.count_nonzero(fp_im), np.count_nonzero(fn_im))

        elif task_type == 'bin_segment':
            for cls_id, cls_name in enumerate(task['categories']):
                gt_bin = gt_im & (1 << cls_id)
                pred_bin = pred_im & (1 << cls_id)
                tp_im = (gt_bin) & (pred_bin)
                fp_im = (~gt_bin) & (pred_bin)
                fn_im = (gt_bin) & (~pred_bin)
                counts['{}_{}_f1'.format(task_name, cls_name)] = (np.count_nonzero(tp_im), np.count_nonzero(fp_im), np.count_nonzero(fn_im))

    return counts

def get_scores(evaluator):
    all_counts = evaluator.map(func=compare)

    sums = {}
    for counts in all_counts:
        for label, (tp, fp, fn) in counts.items():
            if label not in sums:
                sums[label] = [0, 0, 0]
            sums[label][0] += tp
            sums[label][1] += fp
            sums[label][2] += fn

    label_scores = {}
    for label, (tp, fp, fn) in sums.items():
        if tp + fn == 0:
            continue

        # Compute precision and recall, and F1 score.
        if tp == 0:
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        label_scores[label] = f1

    avg_regress_error = np.mean([v for k, v in label_scores.items() if k.endswith('_error')])
    avg_segment_f1 = np.mean([v for k, v in label_scores.items() if k.endswith('_f1')])

    return [
        ('regress_error', avg_regress_error),
        ('segment_f1', avg_segment_f1),
    ], label_scores
