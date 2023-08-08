This document contains examples of applying our models on custom images.

## High-Resolution Inference Example

In this example we will obtain high-resolution satellite or aerial imagery and apply a single-image high-resolution model on it.

We will assume you're using [satlas-model-v1-highres.pth](https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth) (pre-trained on SatlasPretrain).
The expected input is 8-bit RGB image, and input values should be divided by 255 so they are between 0-1.

First, obtain the code and the model:

    git clone https://github.com/allenai/satlas
    mkdir models
    wget -O models/satlas-model-v1-highres.pth https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth

Second, find the longitude and latitude of the location you're interested in, and convert it to a Web-Mercator tile at zoom 16-18. You can use the [Satlas Map](https://satlas.allen.ai/map) and hover your mouse over a point of interest to get its longitude and latitude. To convert to tile using Python:

    import satlas.util
    longitude = -122.333
    latitude = 47.646
    print(satlas.util.geo_to_mercator((longitude, latitude), pixels=1, zoom=17))

Get a high-resolution image that you want to apply the model on, e.g. you could download an image from Google Maps by visiting a URL like this:

    http://mt1.google.com/vt?lyrs=s&x={x}&y={y}&z={z}
    Example: http://mt1.google.com/vt?lyrs=s&x=20995&y=45754&z=17

We'll assume the image is saved as `image.jpg`. Now we will load the model and apply the model, and extract its building predictions:

    import json
    import numpy as np
    import skimage.draw
    import skimage.io
    import torch
    import torchvision

    import satlas.model.model
    import satlas.model.evaluate

    # Locations of model config and weights, and the 8-bit RGB image to run inference on.
    config_path = 'configs/highres_pretrain_old.txt'
    weights_path = 'models/satlas-model-v1-highres.pth'
    image_path = 'image.jpg'
    out_path = 'buildings.png'

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

    # Read image and get instance segmentation building outputs.
    with torch.no_grad():
        im = torchvision.io.read_image(image_path)
        gpu_im = im.to(device).float() / 255
        outputs, _ = model([gpu_im])
        # Get output from building (#14) head and first image.
        outputs = outputs[14][0]
        # Move scores, boxes, and masks to CPU.
        scores = outputs['scores'].cpu().numpy()
        boxes = outputs['boxes'].cpu().numpy()
        masks = outputs['masks'].cpu().numpy()[:, 0, :, :]

    # Visualize high-probability buildings.
    wanted = scores > 0.5
    boxes = boxes[wanted]
    masks = masks[wanted]
    all_polygons = satlas.model.evaluate.polygonize_masks(masks, boxes)
    out_im = np.zeros((im.shape[1], im.shape[2]), dtype=np.uint8)
    for polygon_list in all_polygons:
        for coords in polygon_list:
            exterior = np.array(coords[0], dtype=np.int32)
            rows, cols = skimage.draw.polygon(exterior[:, 1], exterior[:, 0], shape=(out_im.shape[0], out_im.shape[1]))
            out_im[rows, cols] = 255
    skimage.io.imsave(out_path, out_im)

See `configs/highres_pretrain_old.txt` for a list of all the heads of this model.
See also `satlas/model/evaluate.py` for examples of how to visualize outputs from head types other than instance segmentation.


## Sentinel-2 Inference Example

In this example we will download three Sentinel-2 scenes of the same location and apply a multi-image low-resolution model on it.

We will assume you're using the solar farm model ([models/solar_farm/best.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar)) but you could use another model like [satlas-model-v1-lowres-multi.pth](https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-lowres-multi.pth) (the SatlasPretrain model) instead.

First obtain the code and the model:

    git clone https://github.com/allenai/satlas
    cd satlas
    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar
    tar xvf satlas_explorer_datasets_2023-07-24.tar

Use [scihub.copernicus.eu](https://scihub.copernicus.eu/dhus/) to download three Sentinel-2 scenes of the same location.

1. Create an account using the profile icon in the top-right.
2. Zoom in on a location of interest, and use the rectangle tool (middle right, the square with dotted lines icon) to draw a rectangle.
3. Open the filters (three horizontal bars icon) in the top-left, check "Mission: Sentinel-2", and select S2MSI1C for product type. Optionally limit cloud cover to "[0 TO 20]" or similar. Optionally add start/end times under Sensing Period.
4. Press the search button. You should see a list of Sentinel-2 scenes, and when you hover over one of them it should highlight the scene on the map.
5. Find three scenes covering the same geographic extent (based on what's highlighted in the map when you hover over that item in the product list) and download them.

Use gdal to merge the bands across scenes:

    TODO

TODO
