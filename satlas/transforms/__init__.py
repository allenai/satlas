from satlas.transforms.augment import CropFlip, Pad, MixChannels, Noise, Brightness, Resize

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for transform in self.transforms:
            image, targets = transform(image, targets)
        return image, targets

class ComposeBatch(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for transform in self.transforms:
            image, targets = transform.apply_batch(image, targets)
        return image, targets

def get_transform(cfg, transforms_cfg):
    import satlas.transforms as satlas_transforms
    if len(transforms_cfg) == 0:
        return None

    transforms = []
    for transform_cfg in transforms_cfg:
        transform_cls = getattr(satlas_transforms, transform_cfg['Name'])
        transform = transform_cls(cfg, transform_cfg)
        transforms.append(transform)
    return Compose(transforms)

def get_batch_transform(cfg, transforms_cfg):
    import satlas.transforms as satlas_transforms
    if len(transforms_cfg) == 0:
        return None

    transforms = []
    for transform_cfg in transforms_cfg:
        transform_cls = getattr(satlas_transforms, transform_cfg['Name'])
        transform = transform_cls(cfg, transform_cfg)
        transforms.append(transform)
    return ComposeBatch(transforms)
