import math
import random
import torch
import torchvision

import satlas.model.dataset

class CropFlip(object):
    def __init__(self, cfg, my_cfg):
        self.my_cfg = my_cfg
        task_specs = cfg['Tasks']
        self.tasks = [satlas.model.dataset.tasks[spec['Name']] for spec in task_specs]

    def __call__(self, data, targets):
        flip_horizontal = self.my_cfg.get('HorizontalFlip', False) and random.random() < 0.5
        flip_vertical = self.my_cfg.get('VerticalFlip', False) and random.random() < 0.5

        # Crop either by fixed amount or random amount depending on config.
        if 'CropMax' in self.my_cfg and 'CropMin' in self.my_cfg:
            crop_max = self.my_cfg['CropMax']
            crop_min = self.my_cfg['CropMin']
            multiple_of = self.my_cfg.get('MultipleOf', 1)
            crop_to = random.randint(crop_min, crop_max)
            crop_to = (crop_to // multiple_of) * multiple_of
        else:
            crop_to = self.my_cfg.get('Crop', data.shape[1])

        return self.apply_one(data, targets, crop_to=crop_to, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)

    def apply_one(self, data, targets, crop_to=None, flip_horizontal=False, flip_vertical=False):
        if crop_to is None:
            crop_to = data.shape[1]

        if data.shape[2] < crop_to or data.shape[1] < crop_to:
            crop_left, crop_top = 0, 0
        else:
            crop_left = random.randint(0, data.shape[2] - crop_to)
            crop_top = random.randint(0, data.shape[1] - crop_to)

        def crop_and_flip(im):
            if im.shape[-1] < crop_to:
                im = torch.nn.functional.pad(im, [0, crop_to - im.shape[-1], 0, 0])
            if im.shape[-2] < crop_to:
                im = torch.nn.functional.pad(im, [0, 0, 0, crop_to - im.shape[-2]])

            if len(im.shape) == 3:
                im = im[:, crop_top:crop_top+crop_to, crop_left:crop_left+crop_to]
                if flip_horizontal:
                    im = torch.flip(im, dims=[2])
                if flip_vertical:
                    im = torch.flip(im, dims=[1])
                return im
            elif len(im.shape) == 2:
                im = im[crop_top:crop_top+crop_to, crop_left:crop_left+crop_to]
                if flip_horizontal:
                    im = torch.flip(im, dims=[1])
                if flip_vertical:
                    im = torch.flip(im, dims=[0])
                return im

        data = crop_and_flip(data)

        for task_idx in range(len(targets)):
            task = self.tasks[task_idx]

            if task['type'] in ['classification', 'multi-label-classification']:
                pass

            elif task['type'] in ['segment', 'bin_segment', 'regress']:
                targets[task_idx]['im'] = crop_and_flip(targets[task_idx]['im'])
                targets[task_idx]['valid_im'] = crop_and_flip(targets[task_idx]['valid_im'])

            elif task['type'] == 'detect' or task['type'] == 'instance':
                target = targets[task_idx]

                if task['type'] == 'instance':
                    # Update masks here since we want to set the dimensions correct even if there are zero boxes.
                    target['masks'] = crop_and_flip(target['masks'][:, :, :])

                if len(target['boxes']) == 0:
                    continue

                centers = torch.stack([
                    (target['boxes'][:, 0] + target['boxes'][:, 2]) / 2,
                    (target['boxes'][:, 1] + target['boxes'][:, 3]) / 2,
                ], dim=1)
                valid_indices = ((centers[:, 0] > crop_left) &
                    (centers[:, 0] < crop_left+crop_to) &
                    (centers[:, 1] > crop_top) &
                    (centers[:, 1] < crop_top+crop_to))

                if task['type'] == 'instance':
                    valid_indices &= (target['masks'].amax(dim=[1, 2]) > 0)
                    target['masks'] = target['masks'][valid_indices, :, :].contiguous()

                target['boxes'] = target['boxes'][valid_indices, :].contiguous()
                target['labels'] = target['labels'][valid_indices].contiguous()

                target['boxes'][:, 0] -= crop_left
                target['boxes'][:, 1] -= crop_top
                target['boxes'][:, 2] -= crop_left
                target['boxes'][:, 3] -= crop_top

                # Weird special case.
                if len(target['boxes']) == 0:
                    target['labels'] = torch.zeros((1,), dtype=torch.int64)

                if flip_horizontal:
                    target['boxes'] = torch.stack([
                        crop_to - target['boxes'][:, 2],
                        target['boxes'][:, 1],
                        crop_to - target['boxes'][:, 0],
                        target['boxes'][:, 3],
                    ], dim=1)
                if flip_vertical:
                    target['boxes'] = torch.stack([
                        target['boxes'][:, 0],
                        crop_to - target['boxes'][:, 3],
                        target['boxes'][:, 2],
                        crop_to - target['boxes'][:, 1],
                    ], dim=1)

            else:
                raise Exception('CropFlip unhandled task type')

        return data, targets

    def apply_batch(self, images, targets):
        # We can apply CropFlip as a batch transform, in which case we only crop, by a random amount.
        crop_max = self.my_cfg['CropMax']
        crop_min = self.my_cfg['CropMin']
        multiple_of = self.my_cfg.get('MultipleOf', 1)
        crop_to = random.randint(crop_min, crop_max)
        crop_to = (crop_to // multiple_of) * multiple_of

        out_images = []
        out_targets = []
        for i in range(len(images)):
            image, target = self.apply_one(images[i], targets[i], crop_to=crop_to)
            out_images.append(image)
            out_targets.append(target)

        return out_images, out_targets

class Pad(object):
    def __init__(self, cfg, my_cfg):
        self.my_cfg = my_cfg
        self.mode = my_cfg.get('Mode', 'topleft')

        task_specs = cfg['Tasks']
        self.tasks = [satlas.model.dataset.tasks[spec['Name']] for spec in task_specs]

    def __call__(self, data, targets):
        # Crop either by fixed amount or random amount depending on config.
        if 'PadMax' in self.my_cfg and 'PadMin' in self.my_cfg:
            pad_max = self.my_cfg['PadMax']
            pad_min = self.my_cfg['PadMin']
            multiple_of = self.my_cfg.get('MultipleOf', 1)
            pad_to = random.randint(pad_min, pad_max)
            pad_to = (pad_to // multiple_of) * multiple_of
        else:
            pad_to = self.my_cfg.get('PadTo', data.shape[1])

        return self.apply_one(data, targets, pad_to=pad_to)

    def apply_one(self, data, targets, pad_to=None):
        def pad(im):
            if len(im.shape) == 2:
                return pad(im.unsqueeze(0))[0, :, :]

            if self.mode == 'topleft':
                if im.shape[2] < pad_to:
                    im = torch.nn.functional.pad(im, [0, pad_to - im.shape[2], 0, 0])
                else:
                    im = im[:, :, 0:pad_to]

                if im.shape[1] < pad_to:
                    im = torch.nn.functional.pad(im, [0, 0, 0, pad_to - im.shape[1]])
                else:
                    im = im[:, 0:pad_to, :]

            elif self.mode == 'center':
                if im.shape[2] < pad_to:
                    half = (pad_to - im.shape[2]) // 2
                    im = torch.nn.functional.pad(im, [pad_to - im.shape[2] - half, half, 0, 0])
                else:
                    start = (im.shape[2] - pad_to) // 2
                    im = im[:, :, start:start+pad_to]

                if im.shape[1] < pad_to:
                    half = (pad_to - im.shape[1]) // 2
                    im = torch.nn.functional.pad(im, [0, 0, pad_to - im.shape[1] - half, half])
                else:
                    start = (im.shape[1] - pad_to) // 2
                    im = im[:, start:start+pad_to, :]

            else:
                raise Exception('bad pad mode {}'.format(self.mode))

            return im

        data = pad(data)

        for task_idx in range(len(targets)):
            task = self.tasks[task_idx]

            if task['type'] in ['classification', 'multi-label-classification']:
                pass

            elif task['type'] in ['segment', 'bin_segment', 'regress']:
                targets[task_idx]['im'] = pad(targets[task_idx]['im'])
                targets[task_idx]['valid_im'] = pad(targets[task_idx]['valid_im'])

            elif task['type'] in ['detect', 'instance']:
                target = targets[task_idx]

                if task['type'] == 'instance':
                    # Update masks here since we want to set the dimensions correct even if there are zero boxes.
                    target['masks'] = pad(target['masks'][:, :, :])

                if len(target['boxes']) == 0:
                    continue

                centers = torch.stack([
                    (target['boxes'][:, 0] + target['boxes'][:, 2]) / 2,
                    (target['boxes'][:, 1] + target['boxes'][:, 3]) / 2,
                ], dim=1)
                valid_indices = (centers[:, 0] < pad_to) & (centers[:, 1] < pad_to)

                if task['type'] == 'instance':
                    valid_indices &= (target['masks'].amax(dim=[1, 2]) > 0)
                    target['masks'] = target['masks'][valid_indices, :, :].contiguous()

                target['boxes'] = target['boxes'][valid_indices, :].contiguous()
                target['labels'] = target['labels'][valid_indices].contiguous()

                # Weird special case.
                if len(target['boxes']) == 0:
                    target['labels'] = torch.zeros((1,), dtype=torch.int64)

            else:
                raise Exception('Pad unhandled task type')

        return data, targets

class MixChannels(object):
    def __init__(self, cfg, my_cfg):
        # mix_groups is a list of groups of channel ranges that can be swapped among each other.
        # For example, [[(0, 3), (3, 6)]] specifies channels 0-3 and 3-6 can be swapped.
        # While [[(3, 4), (6, 7)], [(4, 5), (7, 8)]] specifies channels 3 and 6 can be swapped, while 4 and 7 can also be swapped.
        # A channel should only appear in one group.
        self.mix_groups = my_cfg['MixGroups']

    def __call__(self, data, targets):
        new_data = torch.clone(data)
        for group in self.mix_groups:
            # Permute within this group.
            src_channels = group
            dst_channels = list(group)
            random.shuffle(dst_channels)

            # Assign the dst channels.
            for src_range, dst_range in zip(src_channels, dst_channels):
                new_data[dst_range[0]:dst_range[1], :, :] = data[src_range[0]:src_range[1], :, :]

        return new_data, targets

class Noise(object):
    def __init__(self, cfg, my_cfg):
        self.noise = my_cfg['Noise']

    def __call__(self, data, targets):
        data = data.float() + self.noise*torch.randn(data.shape)
        data = torch.clip(data, min=0, max=255)
        return data.byte(), targets

class Brightness(object):
    def __init__(self, cfg, my_cfg):
        self.max_factor = my_cfg['MaxFactor']
        self.max_bias = my_cfg['MaxBias']

    def __call__(self, data, targets):
        factor = 1.0 - self.max_factor + 2*self.max_factor*random.random()
        bias = -self.max_bias + 2*self.max_bias*random.random()
        data = data.float() * factor + bias
        data = torch.clip(data, min=0, max=255)
        return data.byte(), targets

# RandomResize only works as a batch transform,
# where all the images in the batch are resized to the same size,
# since otherwise padding would be needed to align the image sizes.
class Resize(object):
    def __init__(self, cfg, my_cfg):
        self.my_cfg = my_cfg
        task_specs = cfg['Tasks']
        self.tasks = [satlas.model.dataset.tasks[spec['Name']] for spec in task_specs]

    def apply_one(self, data, targets, resize_to=None):
        assert data.shape[1] == data.shape[2] # currently we only support square image since we rescale both dimensions by same factor
        factor = resize_to / data.shape[1]
        data = torchvision.transforms.functional.resize(data, [resize_to, resize_to])

        # Helper function for resizing, since unsqueezing is needed only if it's [H, W] only (no channels).
        def resize_label(im):
            if len(im.shape) == 2:
                return torchvision.transforms.functional.resize(
                    img=im.unsqueeze(0),
                    size=[resize_to, resize_to],
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )[0, :, :]
            else:
                return torchvision.transforms.functional.resize(
                    img=im,
                    size=[resize_to, resize_to],
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )

        for task_idx in range(len(targets)):
            task = self.tasks[task_idx]

            if task['type'] in ['classification', 'multi-label-classification']:
                pass

            elif task['type'] in ['segment', 'bin_segment', 'regress']:
                targets[task_idx]['im'] = resize_label(targets[task_idx]['im'])
                targets[task_idx]['valid_im'] = resize_label(targets[task_idx]['valid_im'])

            elif task['type'] == 'detect':
                targets[task_idx]['boxes'] *= factor

            elif task['type'] == 'instance':
                targets[task_idx]['boxes'] *= factor
                if targets[task_idx]['masks'].shape[0] > 0:
                    targets[task_idx]['masks'] = resize_label(targets[task_idx]['masks'])
                else:
                    # If we have zero boxes, then resize will fail so we just create new tensor of the right size.
                    targets[task_idx]['masks'] = torch.zeros((0, resize_to, resize_to), dtype=targets[task_idx]['masks'].dtype, device=targets[task_idx]['masks'].device)

            else:
                raise Exception('Resize unhandled task type')

        return data, targets

    def __call__(self, data, targets):
        resize_max = self.my_cfg['ResizeMax']
        resize_min = self.my_cfg['ResizeMin']
        multiple_of = self.my_cfg.get('MultipleOf', 1)
        resize_to = random.randint(resize_min, resize_max)
        resize_to = (resize_to // multiple_of) * multiple_of
        return self.apply_one(data, targets, resize_to=resize_to)

    def apply_batch(self, images, targets):
        resize_max = self.my_cfg['ResizeMax']
        resize_min = self.my_cfg['ResizeMin']
        multiple_of = self.my_cfg.get('MultipleOf', 1)
        resize_to = random.randint(resize_min, resize_max)
        resize_to = (resize_to // multiple_of) * multiple_of

        out_images = []
        out_targets = []
        for i in range(len(images)):
            image, target = self.apply_one(images[i], targets[i], resize_to=resize_to)
            out_images.append(image)
            out_targets.append(target)

        return out_images, out_targets
