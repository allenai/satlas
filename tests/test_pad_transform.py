import sys
import types
import unittest

import torch

# augment imports this module at import time but the transform does not use it.
sys.modules.setdefault('satlas.model.dataset', types.ModuleType('satlas.model.dataset'))
from satlas.transforms.augment import Pad


class PadDetectionTargetsTest(unittest.TestCase):
    def make_pad(self, mode):
        cfg = {'Tasks': [{'Task': {'type': 'detect'}}]}
        return Pad(cfg, {'Mode': mode})

    def test_center_padding_offsets_boxes_and_keeps_targets_aligned(self):
        image = torch.zeros((1, 4, 6))
        targets = [{'boxes': torch.tensor([[1., 1., 3., 3.]]),
                    'labels': torch.tensor([7])}]

        padded, transformed = self.make_pad('center').apply_one(
            image, targets, pad_to=8)

        self.assertEqual(tuple(padded.shape), (1, 8, 8))
        torch.testing.assert_close(
            transformed[0]['boxes'], torch.tensor([[2., 3., 4., 5.]]))
        torch.testing.assert_close(transformed[0]['labels'], torch.tensor([7]))

    def test_center_crop_offsets_boxes_and_drops_centers_outside_crop(self):
        image = torch.zeros((1, 10, 10))
        targets = [{'boxes': torch.tensor([[1., 1., 3., 3.],
                                           [7., 7., 9., 9.]]),
                    'labels': torch.tensor([4, 9])}]

        cropped, transformed = self.make_pad('center').apply_one(
            image, targets, pad_to=6)

        self.assertEqual(tuple(cropped.shape), (1, 6, 6))
        torch.testing.assert_close(
            transformed[0]['boxes'], torch.tensor([[-1., -1., 1., 1.]]))
        torch.testing.assert_close(transformed[0]['labels'], torch.tensor([4]))


if __name__ == '__main__':
    unittest.main()
