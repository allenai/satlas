import argparse
import json
import numpy as np
import os
import skimage.io
import sys
import time
import torch
import torchvision

import satlas.model.dataset
import satlas.model.model
import satlas.model.util
import satlas.model.evaluate
import satlas.transforms

def main(args):
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Override config with command-line arguments.
    if ('vis_dir' in args) and args.vis_dir is not None:
        config['VisDir'] = args.vis_dir
    if ('probs_dir' in args) and args.probs_dir is not None:
        config['ProbsDir'] = args.probs_dir
    if ('out_dir' in args) and args.out_dir is not None:
        config['OutDir'] = args.out_dir
    if ('split' in args) and args.split is not None:
        for spec in config['Tasks']:
            spec['TestSplit'] = args.split
    if ('image_list' in args) and args.image_list is not None:
        for spec in config['Tasks']:
            spec['ImageList'] = args.image_list
    if ('task' in args) and args.task is not None:
        config['EvaluateTask'] = args.task
    if ('batch_size' in args) and args.batch_size:
        config['BatchSize'] = args.batch_size
    if ('num_images' in args) and args.num_images:
        config['NumImages'] = args.num_images
    if ('max_tiles' in args) and args.max_tiles:
        config['TestMaxTiles'] = args.max_tiles

    channels = config.get('Channels', ['tci', 'fake', 'fake'])
    batch_size = config['BatchSize']
    vis_dir = config.get('VisDir', None)
    probs_dir = config.get('ProbsDir', None)
    out_dir = config.get('OutDir', None)
    half_enabled = config.get('Half', False)

    # Load test data.
    test_data = satlas.model.dataset.Dataset(
        task_specs=config['Tasks'],
        transforms=satlas.transforms.get_transform(config, config.get('TestTransforms', [])),
        channels=channels,
        selected_task=config.get('EvaluateTask', None),
        max_tiles=config.get('TestMaxTiles', None),
        num_images=config.get('NumImages', 1),
        phase='Test',
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=config.get('NumLoaderWorkers', 4),
        collate_fn=satlas.model.util.collate_fn,
        sampler=torch.utils.data.SequentialSampler(test_data),
    )

    print('loaded {} test tiles for inference'.format(len(test_data)))

    # Initialize model and device.
    device = torch.device("cuda")
    model_config = config['Model']
    model = satlas.model.model.Model({
        'config': model_config,
        'channels': channels,
        'tasks': config['Tasks'],
    })

    # Where to save best/last weights
    save_path = config['SavePath']
    save_path = save_path.replace('LABEL', os.path.basename(args.config_path).split('.')[0])

    # Load in weights to the model if specified
    if ('weights' in args) and args.weights:
        state_dict = torch.load(args.weights, map_location=device)
    else:
        state_dict = torch.load(os.path.join(save_path, 'best.pth'), map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    evaluator_params_list = None
    if args.evaluator_params:
        evaluator_params_list = json.loads(args.evaluator_params)

    if args.pick_threshold:
        val_data = satlas.model.dataset.Dataset(
            task_specs=config['Tasks'],
            transforms=satlas.transforms.get_transform(config, config.get('ValTransforms', [])),
            channels=channels,
            selected_task=args.task,
            max_tiles=config.get('ValMaxTiles', None),
            num_images=config.get('NumImages', 1),
            phase='Val',
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=config.get('NumLoaderWorkers', 4),
            collate_fn=satlas.model.util.collate_fn,
            sampler=torch.utils.data.SequentialSampler(val_data),
        )

        print('tuning thresholds...')
        _, _, _, evaluator_params_list = satlas.model.evaluate.evaluate(
            config=config,
            model=model,
            device=device,
            loader=val_loader,
            half_enabled=half_enabled,
            print_details=args.details,
            evaluator_params_list=evaluator_params_list,
        )

    print('evaluating...')
    test_loss, test_losses, test_scores, _ = satlas.model.evaluate.evaluate(
        config=config,
        model=model,
        device=device,
        loader=test_loader,
        half_enabled=half_enabled,
        vis_dir=vis_dir,
        probs_dir=probs_dir,
        out_dir=out_dir,
        print_details=args.details,
        evaluator_params_list=evaluator_params_list,
    )
    print('loss={} losses={} scores={}'.format(test_loss, test_losses, test_scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply model on test set.")
    parser.add_argument("--config_path", help="Configuration file path.")
    parser.add_argument("--vis_dir", help="Override visualization directory.", default=None)
    parser.add_argument("--probs_dir", help="Override probabilities directory.", default=None)
    parser.add_argument("--out_dir", help="Override output directory.", default=None)
    parser.add_argument("--task", help="Only report metrics for this task.", default=None)
    parser.add_argument("--split", help="Override test split.", default=None)
    parser.add_argument("--image_list", help="Override image list.", default=None)
    parser.add_argument("--details", help="Print detailed scores", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--evaluator_params", help="Override evaluator params", default=None)
    parser.add_argument("--pick_threshold", help="Tune threshold on validation set", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--weights", help="Path to weights, override SavePath", default=None)
    parser.add_argument("--batch_size", help="Override batch size.", type=int, default=None)
    parser.add_argument("--num_images", help="Override NumImages.", type=int, default=None)
    parser.add_argument("--max_tiles", help="Override TestMaxTiles.", type=int, default=None)
    args = parser.parse_args()

    main(args)
