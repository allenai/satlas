import argparse
import json
import numpy as np
import os
import sys
import time
import torch
import torchvision
from torch.autograd import Variable

import satlas.model.dataset
import satlas.model.evaluate
import satlas.model.model
import satlas.model.util
import satlas.transforms

def make_warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor, warmup_delay=0):
    def f(x):
        if x < warmup_delay:
            return 1
        if x >= warmup_delay + warmup_iters:
            return 1
        alpha = float(x - warmup_delay) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def save_atomic(state_dict, dir_name, fname):
    tmp_fname = fname+'.tmp'
    torch.save(state_dict, os.path.join(dir_name, tmp_fname))
    os.rename(os.path.join(dir_name, tmp_fname), os.path.join(dir_name, fname))

def main(args, config):
    rank = args.local_rank
    primary = rank is None or rank == 0
    is_distributed = rank is not None

    channels = config.get('Channels', ['tci', 'fake', 'fake'])
    batch_size = config['BatchSize'] // args.world_size
    val_batch_size = config.get('ValBatchSize', config['BatchSize'])

    # Set Task info if needed.
    for spec in config['Tasks']:
        if 'Task' not in spec:
            spec['Task'] = satlas.model.dataset.tasks[spec['Name']]
    if 'ChipSize' in config:
        satlas.model.dataset.chip_size = config['ChipSize']

    train_transforms = satlas.transforms.get_transform(config, config.get('TrainTransforms', [{
        'Name': 'CropFlip',
        'HorizontalFlip': True,
        'VerticalFlip': True,
        'Crop': 256,
    }]))
    batch_transform = satlas.transforms.get_batch_transform(config, config.get('TrainBatchTransforms', []))

    def get_task_transforms(k):
        task_transforms = {}
        for spec in config['Tasks']:
            if k not in spec:
                continue
            task_transforms[spec['Name']] = satlas.transforms.get_transform(config, spec[k])
        if len(task_transforms) == 0:
            return None
        return task_transforms

    # Load train and validation data.
    train_data = satlas.model.dataset.Dataset(
        task_specs=config['Tasks'],
        transforms=train_transforms,
        channels=channels,
        max_tiles=config.get('TrainMaxTiles', None),
        num_images=config.get('NumImages', 1),
        task_transforms=get_task_transforms('TrainTransforms'),
        phase="Train",
    )

    val_data = satlas.model.dataset.Dataset(
        task_specs=config['Tasks'],
        transforms=satlas.transforms.get_transform(config, config.get('ValTransforms', [])),
        channels=channels,
        max_tiles=config.get('ValMaxTiles', None),
        num_images=config.get('NumImages', 1),
        task_transforms=get_task_transforms('ValTransforms'),
        phase="Val",
    )

    print('loaded {} train, {} valid'.format(len(train_data), len(val_data)))

    train_sampler_cfg = config.get('TrainSampler', {'Name': 'random'})
    if train_sampler_cfg['Name'] == 'random':
        train_sampler = torch.utils.data.RandomSampler(train_data)
    elif train_sampler_cfg['Name'] == 'tile_weight':
        with open(train_sampler_cfg['Weights'], 'r') as f:
            tile_weights = json.load(f)
        train_sampler = train_data.get_tile_weight_sampler(tile_weights=tile_weights)
    else:
        raise Exception('unknown train sampler {}'.format(train_sampler_cfg['Name']))

    val_sampler = torch.utils.data.SequentialSampler(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config.get('NumLoaderWorkers', 4),
        collate_fn=satlas.model.util.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=val_batch_size,
        sampler=val_sampler,
        num_workers=config.get('NumLoaderWorkers', 4),
        collate_fn=satlas.model.util.collate_fn,
    )

    # Initialize torch distributed.
    if is_distributed:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Initialize model.
    device = torch.device("cuda")
    model_config = config['Model']
    model = satlas.model.model.Model({
        'config': model_config,
        'channels': channels,
        'tasks': config['Tasks'],
    })

    # Load model if requested.
    if 'RestorePath' in config:
        if primary: print('restoring from', config['RestorePath'])
        state_dict = torch.load(config['RestorePath'], map_location=device)

        # Deal with when model weights are in .pth.tar format.
        if config['RestorePath'].endswith('.pth.tar'):
            state_dict = state_dict['state_dict']

        # By default, don't restore model heads.
        # We only restore head if special RestoreHead key is set.
        if not 'RestoreHead' in config:
            for k in list(state_dict.keys()):
                if k.startswith('heads.'):
                    del state_dict[k]

        # User can specify specific prefixes they would like to restore.
        # We keep the state_dict key if it matches any prefix.
        if 'RestorePrefixes' in config:
            for k in list(state_dict.keys()):
                ok = False
                for prefix in config['RestorePrefixes']:
                    if not k.startswith(prefix):
                        continue
                    ok = True
                    break
                if not ok:
                    del state_dict[k]

        # User can also replace prefixes of some keys with another prefix.
        # This is useful when copying the backbone of a slightly different architecture
        # where the backbone ends up having a different prefix in the model.
        # Example: [["", "backbone."]] would prepend "backbone." to every state_dict key.
        if 'RestoreReplacePrefix' in config:
            for old_prefix, new_prefix in config['RestoreReplacePrefix']:
                for k in list(state_dict.keys()):
                    if not k.startswith(old_prefix):
                        continue
                    new_k = new_prefix + k[len(old_prefix):]
                    state_dict[new_k] = state_dict[k]
                    del state_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if primary and (missing_keys or unexpected_keys):
            print('missing={}; unexpected={}'.format(missing_keys, unexpected_keys))

    # Move model to the correct device.
    model.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Prepare save directory.
    save_path = config['SavePath']
    save_path = save_path.replace('LABEL', os.path.basename(args.config_path).split('.')[0])
    if primary:
        os.makedirs(save_path, exist_ok=True)
        print('saving to', save_path)

    # Construct optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_config = config['Optimizer']
    if optimizer_config['Name'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=optimizer_config['InitialLR'])
    elif optimizer_config['Name'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=optimizer_config['InitialLR'])

    # Freeze parameters if desired.
    unfreeze_iters = None
    if 'Freeze' in config:
        freeze_prefixes = config['Freeze']
        freeze_params = []
        for name, param in model.named_parameters():
            should_freeze = False
            for prefix in freeze_prefixes:
                if name.startswith(prefix):
                    should_freeze = True
                    break
            if should_freeze:
                if primary: print('freeze', name)
                param.requires_grad = False
                freeze_params.append((name, param))
        if 'Unfreeze' in config:
            unfreeze_iters = config['Unfreeze'] // batch_size // args.world_size
            def unfreeze_hook():
                for name, param in freeze_params:
                    if primary: print('unfreeze', name)
                    param.requires_grad = True

    # Configure learning rate schedulers.
    if 'WarmupExamples' in config:
        warmup_iters = config['WarmupExamples'] // batch_size // args.world_size
        warmup_delay_iters = config.get('WarmupDelay', 0) // batch_size // args.world_size
        warmup_lr_scheduler = make_warmup_lr_scheduler(
            optimizer, warmup_iters, 1.0/warmup_iters,
            warmup_delay=warmup_delay_iters,
        )
    else:
        warmup_iters = 0
        warmup_lr_scheduler = None

    lr_scheduler = None

    if 'Scheduler' in config:
        scheduler_config = config['Scheduler']
        if scheduler_config['Name'] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=scheduler_config.get('Factor', 0.1),
                patience=scheduler_config.get('Patience', 2),
                min_lr=scheduler_config.get('MinLR', 1e-5),
                cooldown=scheduler_config.get('Cooldown', 5),
            )

    # Half-precision stuff.
    half_enabled = config.get('Half', False)
    scaler = torch.cuda.amp.GradScaler(enabled=half_enabled)

    # Initialize training loop variables.
    best_score = None

    cur_iterations = 0
    summary_iters = config.get('SummaryExamples', 8192) // batch_size // args.world_size
    summary_epoch = 0
    summary_prev_time = time.time()
    train_losses = [[] for _ in config['Tasks']]
    num_epochs = config.get('NumEpochs', 100)
    num_iters = config.get('NumExamples', 0) // batch_size // args.world_size
    if primary: print('training for {} epochs'.format(num_epochs))

    if 'EffectiveBatchSize' in config:
        accumulate_freq = config['EffectiveBatchSize'] // batch_size // args.world_size
    else:
        accumulate_freq = 1

    model.train()
    for epoch in range(num_epochs):
        if num_iters > 0 and cur_iterations > num_iters:
            break

        if primary: print('begin epoch {}'.format(epoch))

        model.train()
        optimizer.zero_grad()

        for images, targets, info in train_loader:
            cur_iterations += 1

            images = [image.to(device).float()/255 for image in images]

            gpu_targets = [
                [{k: v.to(device) for k, v in target_dict.items()} for target_dict in cur_targets]
                for cur_targets in targets
            ]

            if batch_transform:
                images, gpu_targets = batch_transform(images, gpu_targets)

            if cur_iterations == 1:
                print('input shape:', images[0].shape)

            with torch.cuda.amp.autocast(enabled=half_enabled):
                _, losses = model(images, gpu_targets)

            loss = losses.mean()
            if loss == 0.0:
                loss = Variable(loss, requires_grad = True)

            scaler.scale(loss).backward()

            if cur_iterations == 1 or cur_iterations%accumulate_freq == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            for task_idx in range(len(config['Tasks'])):
                train_losses[task_idx].append(losses[task_idx].item())

            if unfreeze_iters and cur_iterations >= unfreeze_iters:
                unfreeze_iters = None
                unfreeze_hook()

            if warmup_lr_scheduler:
                warmup_lr_scheduler.step()
                if cur_iterations > warmup_delay_iters + warmup_iters + 1:
                    print('removing warmup_lr_scheduler')
                    warmup_lr_scheduler = None

            if cur_iterations%summary_iters == 0:
                train_loss = np.mean(train_losses)
                train_task_losses = [np.mean(losses) for losses in train_losses]

                for losses in train_losses:
                    del losses[:]

                if is_distributed:
                    # Update the learning rate across all distributed nodes.
                    dist_train_loss = torch.tensor(train_loss, dtype=torch.float32, device=device)
                    torch.distributed.all_reduce(dist_train_loss, op=torch.distributed.ReduceOp.AVG)
                    if warmup_lr_scheduler is None:
                        lr_scheduler.step(dist_train_loss.item())
                else:
                    if warmup_lr_scheduler is None:
                        lr_scheduler.step(train_loss)

                # Only evaluate on the primary node (for now).
                if primary:
                    print('begin evaluation')
                    eval_time = time.time()
                    model.eval()
                    val_loss, val_task_losses, val_scores, _ = satlas.model.evaluate.evaluate(
                        config=config,
                        model=model,
                        device=device,
                        loader=val_loader,
                        half_enabled=half_enabled,
                    )
                    val_score = np.mean(val_scores)
                    model.train()

                    print('summary_epoch {}: train_loss={} (losses={}) val_loss={} (losses={}) val={}/{} (scores={}) elapsed={},{} lr={}'.format(
                        summary_epoch,
                        train_loss,
                        train_task_losses,
                        val_loss,
                        val_task_losses,
                        val_score,
                        best_score,
                        val_scores,
                        int(eval_time-summary_prev_time),
                        int(time.time()-eval_time),
                        optimizer.param_groups[0]['lr'],
                    ))

                    summary_epoch += 1
                    summary_prev_time = time.time()

                    # Model saving.
                    if is_distributed:
                        # Need to access underlying model in the DistributedDataParallel so keys aren't prefixed with "module.X".
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    save_atomic(state_dict, save_path, 'last.pth')

                    if np.isfinite(val_score) and (best_score is None or val_score > best_score):
                        save_atomic(state_dict, save_path, 'best.pth')
                        best_score = val_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--config_path", help="Configuration file path.")
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    main(args, config)
