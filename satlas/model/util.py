def collate_fn(batch):
    return tuple(zip(*batch))