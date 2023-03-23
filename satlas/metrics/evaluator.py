import json
import multiprocessing
import os
import tqdm

class Evaluator(object):
    def __init__(
        self,
        gt_path,
        pred_path,
        split_fname,
        lowres_only=False,
        threads=32,
        format='static',
    ):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.lowres_only = lowres_only
        self.threads = threads
        self.format = format

        self.split = set()
        with open(split_fname, 'r') as f:
            for col, row in json.load(f):
                self.split.add((col, row))

    def map(self, func, fname=None, args=[]):
        '''
        Compute output of func(gt_path, pred_path) for each tile in parallel.
        - fname: if provided, only compute outputs for tiles where this filename
          exists in gt_path. Additionally, the gt/pred filenames are passed to
          the func rather than the directory.
        - args: additional arguments to pass to func.
        '''

        # Find tiles.
        jobs = []
        def add_job(cur_gt_path, cur_pred_path):
            if fname is None:
                jobs.append([cur_gt_path, cur_pred_path] + args)
                return

            if not os.path.exists(os.path.join(cur_gt_path, fname)):
                return

            jobs.append([
                os.path.join(cur_gt_path, fname),
                os.path.join(cur_pred_path, fname),
            ] + args)

        for tile_str in os.listdir(self.gt_path):
            parts = tile_str.split('_')
            tile = (int(parts[0]), int(parts[1]))
            if tile not in self.split:
                continue

            if self.format == 'static':
                cur_gt_path = os.path.join(self.gt_path, tile_str)
                cur_pred_path = os.path.join(self.pred_path, tile_str)
                add_job(cur_gt_path, cur_pred_path)

            elif self.format == 'dynamic':
                for event_id in os.listdir(os.path.join(self.gt_path, tile_str)):
                    cur_gt_path = os.path.join(self.gt_path, tile_str, event_id)
                    cur_pred_path = os.path.join(self.pred_path, tile_str, event_id)
                    add_job(cur_gt_path, cur_pred_path)


        p = multiprocessing.Pool(self.threads)
        outputs_gen = p.imap(func, jobs)
        outputs = []
        for output in tqdm.tqdm(outputs_gen, total=len(jobs)):
            outputs.append(output)
        p.close()
        return outputs