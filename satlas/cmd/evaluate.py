import argparse
import importlib

from satlas.metrics.evaluator import Evaluator

parser = argparse.ArgumentParser(description="Compute Satlas metrics.")
parser.add_argument("--gt_path", help="Path to ground truth labels.")
parser.add_argument("--pred_path", help="Path to predictions.")
parser.add_argument("--modality", help="Modality to compute metrics for")
args = parser.parse_args()

evaluator = Evaluator(
    gt_path=args.gt_path,
    pred_path=args.pred_path,
)

modality_module = importlib.import_module('satlas.metrics.' + args.modality)
scores = modality_module.get_scores(evaluator)
print(scores)