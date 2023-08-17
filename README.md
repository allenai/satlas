Satlas: Open AI-Generated Geospatial Data
-----------------------------------------

[Satlas](https://satlas.allen.ai/) aims to provide open AI-generated geospatial data that is highly accurate, available globally, and updated on a frequent (monthly) basis.

For an introduction to Satlas, see https://satlas.allen.ai/.

This repository includes:
- Download the training data, including [SatlasPretrain](SatlasPretrain.md), a large-scale remote sensing dataset.
- Download the pre-trained model weights.
- Training and inference code.
- [Download the AI-generated geospatial data](GeospatialDataProducts.md) for offline analysis.

Satlas super-resolution code is in [another repository](https://github.com/allenai/satlas-super-resolution).


Overview
--------

The [AI-generated geospatial data](GeospatialDataProducts.md) in Satlas is computed by applying deep learning models on [Sentinel-2 satellite imagery](https://sentinel.esa.int/web/sentinel/missions/sentinel-2), which is open imagery released by the European Space Agency.

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

See https://satlas-pretrain.allen.ai/ for more information on SatlasPretrain, or [download the dataset and pre-trained models](SatlasPretrain.md).

### Task-Specific Labels and Model Weights

The fine-tuning training data and model weights can be downloaded at https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/satlas_explorer_datasets_2023-07-24.tar.

This download link contains an archive with four folders:
- `base_models/` contains models trained on SatlasPretrain that are used as initialization for fine-tuning.
- `labels/` contains the fine-tuning task-specific training data.
- `models/` contains the trained model weights.
- `splits/` contains metadata about the training and validation splits.

See [Using the Code](#using-the-code) below for details on training and applying models.

The format of the task-specific datasets is described in [DatasetSpec.md](DatasetSpec.md).

The models are trained to make predictions from multiple Sentinel-2 images.
They first extract features from each image independently through a Swin Transformer.
They then apply temporal max pooling on corresponding feature maps at each of four resolutions.
The pooled feature maps are then passed to task-specific heads to make predictions.
See [ModelArchitecture.md](ModelArchitecture.md) for more details.


AI-Generated Geospatial Data
----------------------------

The AI-generated geospatial data in Satlas [can be downloaded here](GeospatialDataProducts.md) for offline analysis.

We have evaluated the accuracy of each model in terms of their precision and recall on each continent. [View the Data Validation Report here.](DataValidationReport.md)


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

### Inference on Custom Images

[See guide on applying Satlas/SatlasPretrain models on custom images.](CustomInference.md#sentinel-2-inference-example)

Contact
-------

If you have feedback about the code, data, or models, or if you would like to see new types of geospatial data that are feasible to produce from Sentinel-2 imagery,
you can contact us by [opening an issue](https://github.com/allenai/satlas/issues/new) or via e-mail at satlas@allenai.org.