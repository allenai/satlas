This is a summary of ongoing projects on the Satlas team at AI2.
We plan to update it roughly on a monthly basis.

Note that all timelines are estimates, and as projects develop there may be delays or we may find that they are infeasible altogether.

## Recent Updates

- 16 February 2024: added this page.
- 6 February 2024: released the [`satlaspretrain_models` package] to easily apply SatlasPretrain foundation models for downstream tasks.


## SatlasPretrain Ease-of-use (ongoing, 75%)

Make it easier for users to apply SatlasPretrain foundation models for downstream tasks.

- So far we have released the [`satlaspretrain_models` package](https://github.com/allenai/satlaspretrain_models/) and [accompanying tutorial](https://github.com/allenai/satlaspretrain_models/blob/main/demo.ipynb).
- We plan to make the models available in [TorchGeo](https://github.com/microsoft/torchgeo) too (expected March 2024).
- We are also working to make the dataset available on Huggingface (expected March 2024). Currently it can be downloaded for free from an S3 bucket but some users have slow download speed.


## satlas.allen.ai 2023 Data (ongoing, 75%)

Add 2023 data to [satlas.allen.ai](https://satlas.allen.ai/).
This has previously been delayed as we observed reduced model accuracy at high latitudes and worked to improve the models.

- Renewable energy infrastructure and marine infrastructure data can already be downloaded through 2023-08 [from the data products page](https://github.com/allenai/satlas/blob/main/GeospatialDataProducts.md).
- Release of data through 2024-01 is ongoing (expected February 2024).
- Updated website is ongoing (expected March 2024).


## Release Updated Training Data and Fine-tuned Models (not started)

We frequently iterate on the tree cover, marine infrastructure, and renewable energy infrastructure training data and models.
We last released these on July 2023 so they are out-of-date.
We need to find a better way to automatically share the training data and models when they are updated.


## Off-shore Platform Classification (exploratory)

Users have asked for classification of detected platforms, e.g. oil and gas platform vs air force tower vs power line.
We plan to release this in Q3 2024.


## Improved Super-Resolution (exploratory)

We are exploring improvements to the accuracy of the super-resolution outputs.
Once the model is improved, we will release the dataset and model, and refresh the super-resolution layer in [satlas.allen.ai](https://satlas.allen.ai/).


## Release Geospatial Data on Google Earth Engine (not started)

We plan to make all of the [geospatial data products](https://github.com/allenai/satlas/blob/main/GeospatialDataProducts.md) available on Google Earth Engine by Q3 2024.


## Land Cover Monitoring and Change Detection (exploratory)

We are exploring the feasibility of global monitoring of land cover and detecting land cover changes (e.g. urban expansion, forest conversion to agriculture) using Sentinel-2 images.
We hope to release this in Q1 2025.


## Forest Loss Cause Classification (ongoing, 25%)

We are developing a model to classify the cause of detected forest loss (e.g. mining, agriculture, hurricane) from Sentinel-2 images.
We hope to release this in Q4 2024.
