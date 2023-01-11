import multiprocessing
import os
import tqdm

class Evaluator(object):
    def __init__(
        self,
        gt_path,
        pred_path,
        lowres_only=False,
        threads=32,
    ):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.lowres_only = lowres_only
        self.threads = threads

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
        for tile_str in os.listdir(self.gt_path):
            for event_id in os.listdir(os.path.join(self.gt_path, tile_str)):
                cur_gt_path = os.path.join(self.gt_path, tile_str, event_id)
                cur_pred_path = os.path.join(self.pred_path, tile_str, event_id)

                if fname is None:
                    jobs.append([cur_gt_path, cur_pred_path] + args)
                    continue

                if not os.path.exists(os.path.join(cur_gt_path, fname)):
                    continue

                jobs.append([
                    os.path.join(cur_gt_path, fname),
                    os.path.join(cur_pred_path, fname),
                ] + args)

        p = multiprocessing.Pool(self.threads)
        outputs_gen = p.imap(func, jobs)
        outputs = []
        for output in tqdm.tqdm(outputs_gen, total=len(jobs)):
            outputs.append(output)
        p.close()
        return outputs