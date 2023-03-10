Satlas: Multi-Task Remote Sensing Dataset
-----------------------------------------

[[Project Website](https://satlas.allen.ai/) | [Arxiv Paper](https://arxiv.org/abs/2211.15660) | [Github](https://github.com/allenai/satlas/)]

This repository includes documentation for the Satlas dataset, along with code to evaluate output accuracy and dataset loading examples.


Download
--------

### Satlas Dataset

The NAIP images, Sentinel-2 images, and labels in Satlas can be downloaded from these respective URLs:

- https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v0-beta-naip.zip
- https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v0-beta-sentinel2.tar
- https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v0-beta-labels.zip

The current release of Satlas is an initial beta release.
As of March 2023, we have expanded the dataset (v1) with more images at each labeled tile (8-12 Sentinel-2 images in 2022 and 3-5 NAIP images in 2011-2020), and it will be released by 1 June 2023.

### Model Weights

Pre-trained weights for single-image SatlasNet can be downloaded from these URLs:

- High-resolution (NAIP + others @ 0.5-2 m/pixel): https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth
- Low-resolution (Sentinel-2): https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-lowres.pth

The model code is not released yet but the Swin-v2-Base backbone can be restored for application to downstream tasks:

    import torch
    import torchvision
    model = torchvision.models.swin_transformer.swin_v2_b()
    full_state_dict = torch.load('satlas-model-v1-highres.pth')
    swin_prefix = 'backbone.backbone.'
    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    model.load_state_dict(swin_state_dict)

The channels should be in RGB order, with pixel values normalized to 0-1.

- NAIP: remove the IR channel if any, and divide the 0-255 RGB values by 255.
- Other high-resolution e.g. Google Earth: divide the 0-255 RGB values by 255.
- Sentinel-2: use the TCI (true-color image) file and divide the 0-255 RGB values by 255.

### Ancillary Datasets

You can also download the ancillary datasets below.
These are part of the broader Satlas project at AI2 but not part of the "Satlas dataset".
Subsets of some of these datasets may appear in Satlas.

- Marine Infrastructure (Vessels, Off-Shore Wind Turbines, Off-Shore Platforms): [[Documentation](marine_infrastructure_dataset.md) | [Download](https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-marine-infrastructure-v1.zip)]


Dataset Structure
-----------------

Satlas is divided into train, validation, and test splits, but the train and test splits have further subdivisions:

- train_gold contains more complete labels for 13 classes: chimney, fountain, gas_station, helipad, parking_garage, parking_lot, petroleum_well, power_substation, silo, storage_tank, toll_booth, water_tower, wind_turbine.
- test_highres focuses evaluation on areas where high-resolution images are available.

For each split, two folders contain the images and labels respectively, for example:

- `satlas/train/`: contains images. `satlas/train/images/` contains Sentinel-2 images and `satlas/train/highres/` contains NAIP images.
- `satlas/train_labels/`: contains labels.


### Tile System

Images and labels are both projected to [Web-Mercator](https://en.wikipedia.org/wiki/Web_Mercator_projection) and stored at zoom level 13.
This means the world is divided into a grid with 2^13 rows and 2^13 columns.
The high-resolution images are stored at zoom level 17.


### Images

The image directory structure looks like this:

    satlas/train/
        images/
            000b0417499f43ffb8907b110d9793d6/
                tci/
                    1867_3287.png
                    1867_3288.png
                    ...
                b05/
                b06/
                ...
            000d5719cfbb4590aa9156fd49d69806/
                ...
            ...
        highres/
            ...

Each folder at the third level contains a different remote sensing image, named with a random UUID like `000b0417499f43ffb8907b110d9793d6`.

The tci, b05, and other sub-folders contain different bands from the image. `tci` contains true-color image; Sentinel-2 includes [other bands](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial) as well.

So `satlas/train/images/abcdef/tci/1867_3287.png` is a 512x512 image corresponding to column 1867 and row 3287 in the grid, for the tci channel of image `abcdef`.
`1867_3288.png` is the image at the next row down.


### Coordinates

The `geo_to_mercator` and `mercator_to_geo` functions in `satlas.util` translate from tile coordinates like (1867, 3287) to longitude-latitude coordinates like (33.4681, -97.9541).

    import satlas.util
    satlas.util.mercator_to_geo((1867, 3287), pixels=1, zoom=13)


### Labels

The label directory structure looks like this:

    satlas/train_labels/
        1434_3312/
            label_0/
            airplane_325/
            coast_3376/
            ...
        1434_3313/
        ...

Each folder at the second level contains labels for a different tile, e.g. `1434_3312` contains labels for the zoom 13 tile at column 1434 and row 3312.

The `label_0`, `airplane_325`, and `coast_3376` folders contain labels for different validity periods (time ranges). The validity period is specified under the metadata key in `airplane_325/vector.json`:

    {
        "metadata": {
            "Start": "2014-08-04T23:00:00+00:00",
            "End": "2014-08-05T01:00:00+00:00"
        },
        "airplane": [
            ...
        ]
    }

This means we think any labels stored here are present in the world from 04 August 2014 23:00 to 05 August 2014 01:00.

Point, polygon, polyline, property, and classification labels are also stored in `vector.json`. For example:

    1234_5678/label_0/vector.json
    {
        "airplane": [{
            "Geometry": {
                "Type": "point",
                "Point": [454, 401]
            },
            "Properties": {}
        }],

        "power_plant": [{
            "Geometry": {
                "Type": "polygon",
                "Polygon": [[[6197, 6210], [6151, 6284], [6242, 6341], [6260, 6312], [6288, 6268], [6197, 6210]]]
            },
            "Properties": {"plant_type": "gas"}
        }],

        "road": [{
            "Geometry": {
                "Type": "polyline",
                "Polyline": [[7287, 789], [7281, 729], [7280, 716], [7277, 690], [7267, 581], [7256, 466], [7242, 312], [7206, -45], [7149, -628]]
            },
            "Properties": {"road_type": "residential"}
        }],

        "smoke": [{
            "Geometry": {
                "Type": "polygon",
                "Polygon": [[[0, 0], [8192, 0], [8192, 8192], [0, 8192]]]
            },
            "Properties": {"smoke": 0}
        }]
    }

The `vector.json` labels contain geometries that describe coordinates within the tile.
We define an 8192x8192 coordinate system within each tile.
These coordinates can be converted to longitude-latitude coordinates like this:

    import satlas.util
    tile = (1234, 5678)
    point = (454, 401)
    satlas.util.mercator_to_geo((tile[0]*8192 + point[0], tile[1]*8192 + point[1]), pixels=8192, zoom=13)

Points contain a single center column and row.

Polygons contain a sequence of rings, each of which is a sequence of points. The first ring is the exterior of the polygon, and any subsequent rings are interior holes.

Polylines just contain a sequence of points.

Classification labels are polygons covering the entire tile, and the category is stored in the property (the category is 0 for smoke above).

Segmentation and regression labels are stored in auxiliary greyscale 512x512 image files.
The task is specified by the image filename, e.g. `water_event.png` or `land_cover.png`.
The tasks and categories are defined in `satlas/metrics/raster.py`.

For regression tasks, the value in the greyscale PNG at a pixel is proportional to the quantity (for tree cover, 0 is no trees and 200 is full tree cover; for digital elevation model, 0 is -20 m water depth and 255 is land).

For segmentation tasks, 0 represents invalid pixels, 1 represents the first cateogry in `satlas/metrics/raster.py`, and so on.

For binary segmentation tasks, the rightmost bit in the greyscale value corresponds to the first category, and so on.

Note that only a subset of categories are annotated in each label folder. Oftentimes categories will be annotated but have no instances present in the tile and/or time range, in which case they will appear in `vector.json` like this:

    "power_substation": [],

If the category is not annotated at all, then it will omit the key in `vector.json` entirely (or, for segmentation and regression labels, omit the PNG image like no `land_cover.png`).


### Other Files

Additional files in `satlas/metadata/` contain extra data.

- `train.json`, `val.json`, `test.json`, `train_gold.json`, and `test_highres.json` enumerate the tiles assigned to each split. Note that `train.json` is a superset of `train_gold.json`, and `test.json` is a superset of `test_highres.json`.
- `image_times_sentinel2.json` and `image_times_naip.json` contain the timestamp of images based on their UUIDs.


Inference
---------

The output format is essentially identical to the format of labels in Satlas. For each label folder like `labels/1434_3312/airplane_325/`, a corresponding output `outputs/1434_3312/airplane_325/` should be produced.


### Predicting non-Property Labels

For predicting label types other than properties, the following data from the labels folder can be used:

- Tile position (e.g. `1434_3312`)
- Validity period (from `vector.json`)
- The subset of categories annotated in the label folder (must only be used for optimizing inference execution)
- Label folder name (must only be used for creating identically named output folder)


### Predicting Property and Classification Labels

For property and classification labels, the entirety of `vector.json` except the property values can also be used.

Thus, the model can use the coordinates of points, polygons, and polylines like `power_plant` or `road`, and optimize execution based on the property keys (e.g. don't predict road width if it's not labeled for a certain road).

The output should be a new version of `vector.json` with the same features but with the property values filled in based on the model predictions.


### Evaluation

    python -m satlas.cmd.evaluate --gt_path path/to/satlas/test_highres_labels/ --pred_path path/to/outputs/ --modality point
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/test_highres_labels/ --pred_path path/to/outputs/ --modality polygon
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/test_highres_labels/ --pred_path path/to/outputs/ --modality polyline
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/test_highres_labels/ --pred_path path/to/outputs/ --modality property
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/test_highres_labels/ --pred_path path/to/outputs/ --modality raster


### Example Visualization Code

[See here.](satlas/cmd/to_dataset/README.md)
