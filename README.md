Satlas: Open AI-Generated Geospatial Data
-----------------------------------------

[[Website](https://satlas.allen.ai/) | [SatlasPretrain](https://satlas-pretrain.allen.ai/) | [Github](https://github.com/allenai/satlas/)]

Satlas aims to provide open AI-generated geospatial data that is highly accurate, available globally, and updated on a frequent (monthly) basis.
For an introduction to Satlas, see https://satlas.allen.ai/.

This repository includes:
- Download the training data, including [SatlasPretrain](SatlasPretrain.md), a large-scale remote sensing dataset.
- Download the pre-trained model weights.
- Training and inference code.
- [Download the AI-generated geospatial data](GeospatialDataProducts.md) for offline analysis.

For Satlas super-resolution code, see https://github.com/allenai/satlas-super-resolution


Overview
--------

The AI-generated geospatial data in Satlas is computed by applying deep learning models on [Sentinel-2 satellite imagery](https://sentinel.esa.int/web/sentinel/missions/sentinel-2), which is open imagery released by the European Space Agency.

The images are relatively low-resolution, at 10 m/pixel, but captured frequently---the bulk of Earth's land mass is imaged weekly by Sentinel-2. We retrieve these images and update the geospatial data products on a monthly basis.


Training Data and Models
------------------------

The models in Satlas are developed in four phases:

1. Pre-train models on SatlasPretrain.
2. Annotate high-quality task-specific training labels.
3. Fine-tune models on the task-specific labels.
4. Test the models on the whole world, and iterate on the training data until the models provide high accuracy.

### SatlasPretrain

SatlasPretrain is a large-scale remote sensing image understanding dataset appearing in ICCV 2023.
It contains 302M labels under 137 categories, collected through a combination of crowdsourced annotation and processing existing data sources like OpenStreetMap.

Pre-training on SatlasPretrain helps to improve the downstream performance of our models when fine-tuning on the smaller sets of task-specific labels.

See https://satlas-pretrain.allen.ai/ for more information on SatlasPretrain, or [download the dataset](SatlasPretrain.md).

### Task-Specific Labels and Model Weights

The fine-tuning training data and model weights can be downloaded at https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar.

This download link contains an archive with four folders:
- `base_models/` contains models trained on SatlasPretrain that are used as initialization for fine-tuning.
- `labels/` contains the fine-tuning training data.
- `models/` contains the trained model weights.
- `splits/` contains metadata about the training and validation splits.

Each training dataset is structured like this:

    labels/solar_farm/
        4267_2839/
            gt.png
            images/
                fp03-2020-12/
                    tci.png
                    b05.png
                    b06.png
                    b07.png
                    b08.png
                    b11.png
                    b12.png
                fp03-2021-01/
                    ...
                ...
        ...

Each folder like `4267_2839/` contains a different training example, which corresponds to a particular geographic location. The `images/` subfolder contains images captured at different times at that location. The current datasets all use [Sentinel-2 images](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). `tci.png` contains B02, B03, and B04. The 10 m/pixel and 20 m/pixel bands (except B8A) are used as input and included in the training data, while the 60 m/pixel bands are not used.

The Sentinel-2 bands were normalized to 8-bit PNGs as follows:
- `tci`: taken from the TCI JPEG2000 image provided by ESA. This is already an 8-bit RGB product.
- Other bands: the raw image scenes from ESA are 16-bit products. We convert to greyscale 8-bit PNGs: `clip(band/32, 0, 255)`.

For segmentation (solar farm) and regression (tree cover) labels, `gt.png` contains a greyscale mask. For segmentation, the pixel value indicates the class ID. For regression, the pixel value indicates the ground truth value.

For object detection labels (on-shore wind turbines and marine infrastructure), `gt.json` contains bounding box labels like this:

    [
        [14, 467, 54, 507, "wind_turbine"],
        [53, 473, 93, 513, "wind_turbine"]
    ]

Each box is in the form `[start_col, start_row, end_col, end_row, category_name]`. The current tasks are annotated as points so the boxes are all the same size, the center point is the actual label and can be easily derived `[(start_col + end_col) / 2, (start_row + end_row) / 2]` if desired.


AI-Generated Geospatial Data
----------------------------

The AI-generated geospatial data in Satlas [can be downloaded here](GeospatialDataProducts.md) for offline analysis.

We have evaluated the accuracy of each model in terms of their precision and recall on each continent. [View the accuracy report here.](AccuracyReport.md)


Using the Code
--------------

Here we describe using the code for the task-specific training data. For using the code for pre-training models on SatlasPretrain, [click here](SatlasPretrain.md).

### Training and Validation

First clone this repository and extract the training data to a subfolder called `satlas_explorer_datasets`:

    git clone https://github.com/allenai/satlas
    cd satlas
    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar
    tar xvf satlas_explorer_datasets_2023-07-24.tar

Run training if desired (this will overwrite the models extracted from the tar download):

    python -m satlas.cmd.model.train --config_path configs/satlas_explorer_wind_turbine.txt
    python -m satlas.cmd.model.train --config_path configs/satlas_explorer_solar_farm.txt
    python -m satlas.cmd.model.train --config_path configs/satlas_explorer_marine_infrastructure.txt
    python -m satlas.cmd.model.train --config_path configs/satlas_explorer_tree_cover.txt

Compute precision and recall stats on the validation data:

    python -m satlas.cmd.model.infer --config_path configs/satlas_explorer_wind_turbine.txt --details
    python -m satlas.cmd.model.infer --config_path configs/satlas_explorer_solar_farm.txt --details
    python -m satlas.cmd.model.infer --config_path configs/satlas_explorer_marine_infrastructure.txt --details
    python -m satlas.cmd.model.infer --config_path configs/satlas_explorer_tree_cover.txt --details

Inference on Custom Images
--------------------------

### High-Resolution Inference Example

In this example we will obtain high-resolution satellite or aerial imagery and apply a single-image high-resolution model on it.

We will assume you're using [satlas-model-v1-highres.pth](https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth) (pre-trained on SatlasPretrain).

First, obtain the code and the model:

    git clone https://github.com/allenai/satlas
    mkdir models
    wget -O models/satlas-model-v1-highres.pth https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-highres.pth

Second, find the longitude and latitude of the location you're interested in, and convert it to a Web-Mercator tile at zoom 16-18. You can use [Satlas](https://satlas.allen.ai/map) and hover your mouse over a point of interest to get its longitude and latitude. To convert to tile using Python:

    import satlas.util
    longitude = -122.333
    latitude = 47.646
    print(satlas.util.geo_to_mercator((longitude, latitude), pixels=1, zoom=18))

Get a high-resolution image that you want to apply the model on, e.g. you could download an image from Google Maps by visiting a URL like this:

    http://mt1.google.com/vt?lyrs=s&x={x}&y={y}&z={z}
    Example: http://mt1.google.com/vt?lyrs=s&x=41991&y=91508&z=18

We'll assume the image is saved as `image.png`. Now we will load the model and apply the model, and extract its building predictions:

    TODO


### Sentinel-2 Inference Example

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
