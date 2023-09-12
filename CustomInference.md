This document contains examples of applying our models on custom images.

If you're fine-tuning models on downstream tasks, check the [Extracting Representations Example](#extracting-representations-example), which shows how to use the pre-trained backbones independently of this codebase.

If you want to compute model outputs, check the [High-Resolution](#high-resolution-inference-example) or [Sentinel-2](#sentinel-2-inference-example) inference example.

## Extracting Representations Example

In this example we will load pre-trained backbone without using this codebase.
If you want to use our model architecture code, then the other examples below may be more helpful.

Here's code to restore the Swin-v2-Base backbone of a single-image model for application to downstream tasks:

    import torch
    import torchvision
    model = torchvision.models.swin_transformer.swin_v2_b()
    full_state_dict = torch.load('satlas-model-v1-highres.pth')
    # Extract just the Swin backbone parameters from the full state dict.
    swin_prefix = 'backbone.backbone.'
    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    model.load_state_dict(swin_state_dict)

See [Normalization.md](Normalization.md) for documentation on how images should be normalized for input to Satlas models.

Feature representations can be extracted like this:

    # Assume im is shape (C, H, W).
    x = im[None, :, :, :]
    outputs = []
    for layer in model.features:
        x = layer(x)
        outputs.append(x.permute(0, 3, 1, 2))
    map1, map2, map3, map4 = outputs[-7], outputs[-5], outputs[-3], outputs[-1]

Here's code to compute the feature representations from a multi-image model through max temporal pooling. Note the different prefix of the Swin backbone parameters. [See here for model architecture details.](ModelArchitecture.md)

    import torch
    import torchvision
    model = torchvision.models.swin_transformer.swin_v2_b()
    # Make sure to load a multi-image model here.
    # Only the multi-image models are trained to provide robust features after max temporal pooling.
    full_state_dict = torch.load('satlas-model-v1-lowres-multi.pth')
    # Extract just the Swin backbone parameters from the full state dict.
    swin_prefix = 'backbone.backbone.backbone.'
    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    model.load_state_dict(swin_state_dict)

    # Assume im is shape (N, C, H, W), with N aligned images of the same location at different times.
    # First get feature maps of each individual image.
    x = im
    outputs = []
    for layer in model.features:
        x = layer(x)
        outputs.append(x.permute(0, 3, 1, 2))
    feature_maps = [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]
    # Now apply max temporal pooling.
    feature_maps = [
        m.amax(dim=0)
        for m in feature_maps
    ]
    # feature_maps can be passed to a head, and the head or entire model can be trained to fine-tune on task-specific labels.

## High-Resolution Inference Example

In this example we will apply a single-image high-resolution model on a high-resolution image.

If you don't have an image already, [see an example of obtaining one](Normalization.md#high-resolution-images).
We will assume the image is saved as `image.jpg`.`

We will assume you're using [satlas-model-v1-highres.pth](https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth) (pre-trained on SatlasPretrain).
The expected input is 8-bit RGB image, and input values should be divided by 255 so they are between 0-1.

First, obtain the code and the model:

    git clone https://github.com/allenai/satlas
    mkdir models
    wget -O models/satlas-model-v1-highres.pth https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth

Load the model and apply the model, and extract its building predictions:

    import json
    import skimage.io
    import torch
    import torchvision

    import satlas.model.model
    import satlas.model.evaluate

    # Locations of model config and weights, and the 8-bit RGB image to run inference on.
    config_path = 'configs/highres_pretrain_old.txt'
    weights_path = 'models/satlas-model-v1-highres.pth'
    image_path = 'image.jpg'

    # Read config and initialize the model.
    with open(config_path, 'r') as f:
        config = json.load(f)
    device = torch.device("cuda")
    for spec in config['Tasks']:
        if 'Task' not in spec:
            spec['Task'] = satlas.model.dataset.tasks[spec['Name']]
    model = satlas.model.model.Model({
        'config': config['Model'],
        'channels': config['Channels'],
        'tasks': config['Tasks'],
    })
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Read image, apply model, and save output visualizations.
    with torch.no_grad():
        im = torchvision.io.read_image(image_path)
        gpu_im = im.to(device).float() / 255
        outputs, _ = model([gpu_im])

        for task_idx, spec in enumerate(config['Tasks']):
            vis_output, _, _, _ = satlas.model.evaluate.visualize_outputs(
                task=spec['Task'],
                image=im.numpy().transpose(1, 2, 0),
                outputs=outputs[task_idx][0],
                return_vis=True,
            )
            if vis_output is not None:
                skimage.io.imsave('out_{}.png'.format(spec['Name']), vis_output)

See `configs/highres_pretrain_old.txt` for a list of all the heads of this model.

## Sentinel-2 Inference Example

In this example we will apply a multi-image multi-band Sentinel-2 model on Sentinel-2 imagery.

If you don't have Sentinel-2 images merged and normalized for Satlas already, [see the example](Normalization.md#sentinel-2-images).
The example also documents the normalization of Sentinel-2 bands expected by our models.

We will assume you're using the solar farm model ([models/solar_farm/best.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar)) but you could use another model like [satlas-model-v1-lowres-multi.pth](https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-lowres-multi.pth) (the SatlasPretrain model) instead.

First obtain the code and the model:

    git clone https://github.com/allenai/satlas
    cd satlas
    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar
    tar xvf satlas_explorer_datasets_2023-07-24.tar

Now we can load the images, normalize them, and apply the model:

    import json
    import numpy as np
    from osgeo import gdal
    import skimage.io
    import torch
    import torchvision
    import tqdm

    import satlas.model.evaluate
    import satlas.model.model

    # Locations of model config and weights, and the input image.
    config_path = 'configs/satlas_explorer_solar_farm.txt'
    weights_path = 'satlas_explorer_datasets/models/solar_farm/best.pth'
    image_path = 'stack.npy'

    # Read config and initialize the model.
    with open(config_path, 'r') as f:
        config = json.load(f)
    device = torch.device("cuda")
    for spec in config['Tasks']:
        if 'Task' not in spec:
            spec['Task'] = satlas.model.dataset.tasks[spec['Name']]
    model = satlas.model.model.Model({
        'config': config['Model'],
        'channels': config['Channels'],
        'tasks': config['Tasks'],
    })
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image = np.load(image_path)
    # For (N, C, H, W) image (with N timestamps), convert to (N*C, H, W).
    image = image.reshape(image.shape[0]*image.shape[1], image.shape[2], image.shape[3])

    # The image is large so apply it on windows.
    # Here we collect outputs from head 0 which is the only head of the solar farm model.
    vis_output = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)
    crop_size = 1024
    head_idx = 0

    with torch.no_grad():
        for row in tqdm.tqdm(range(0, image.shape[1], crop_size)):
            for col in range(0, image.shape[2], crop_size):
                crop = image[:, row:row+crop_size, col:col+crop_size]
                vis_crop = crop.transpose(1, 2, 0)[:, :, 0:3]
                gpu_crop = torch.as_tensor(crop).to(device).float() / 255
                outputs, _ = model([gpu_crop])
                vis_output_crop, _, _, _ = satlas.model.evaluate.visualize_outputs(
                    task=config['Tasks'][head_idx]['Task'],
                    image=vis_crop,
                    outputs=outputs[head_idx][0],
                    return_vis=True,
                )
                if len(vis_output_crop.shape) == 2:
                    vis_output[row:row+crop_size, col:col+crop_size, 0] = vis_output_crop
                    vis_output[row:row+crop_size, col:col+crop_size, 1] = vis_output_crop
                    vis_output[row:row+crop_size, col:col+crop_size, 2] = vis_output_crop
                else:
                    vis_output[row:row+crop_size, col:col+crop_size, :] = vis_output_crop

    skimage.io.imsave('rgb.png', image[0:3, :, :].transpose(1, 2, 0))
    skimage.io.imsave('output.png', vis_output)
