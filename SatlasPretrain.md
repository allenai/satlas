SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding (ICCV 2023)
----------------------------------------------------------------------------------------

[[SatlasPretrain Website](https://satlas-pretrain.allen.ai/) | [Paper](https://arxiv.org/abs/2211.15660) | [Supplementary Material](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/SatlasPretrain_supplementary.pdf) | [Satlas Website](https://satlas.allen.ai/)]

SatlasPretrain is a large-scale pre-training dataset for remote sensing image understanding.

It consists of 302M labels under 137 categories and seven label types: points, polygons, polylines, properties, segmentation labels, regression labels, and classification labels.

Pre-training on SatlasPretrain increases average performance on seven downstream tasks by 18% over ImageNet and 6% over [DOTA](https://captain-whu.github.io/DOTA/index.html) and [iSAID](https://captain-whu.github.io/iSAID/).

Both the dataset and pre-trained models weights can be downloaded below.

Download
--------

### SatlasPretrain Dataset

Satlas consists of Sentinel-2 images, NAIP images, corresponding labels, and metadata which can be downloaded as follows:

    cd satlas_root/
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-sentinel2-a.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-sentinel2-b.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-sentinel1.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2011.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2012.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2013.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2014.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2015.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2016.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2017.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2018.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2019.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-2020.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-labels-dynamic.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-labels-static.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-metadata.tar
    ls | grep tar | xargs -L 1 tar xvf

Small versions of the NAIP and Sentinel-2 images are available. These can be used in conjunction with the labels and metadata above.

    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-small.tar
    wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-sentinel2-small.tar
    ls | grep tar | xargs -L 1 tar xvf
    ln -s sentinel2_small sentinel2
    ln -s naip_small naip

Although not part of the original paper, we have also prepared Landsat and Sentinel-1 images in 2022 corresponding to the static labels in the dataset:

    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/satlas-dataset-v1-landsat.tar
    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/satlas-dataset-v1-sentinel1.tar

SatlasPretrain includes images and labels from the sources below, re-distributed under the original licenses.
For a complete breakdown, see pg 4-6 of the [supplementary material](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/SatlasPretrain_supplementary.pdf).

- [Sentinel-1, Sentinel-2](https://sentinel.esa.int/web/sentinel/missions) (ESA): see https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice
- [NAIP](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip) (USGS): public domain
- [OpenStreetMap](https://www.openstreetmap.org): ODbL
- [C2S](https://mlhub.earth/data/c2smsfloods_v1): CC-BY-4.0
- [Microsoft Buildings](https://github.com/microsoft/USBuildingFootprints): ODbL
- [WorldCover](https://esa-worldcover.org/): CC-BY-4.0
- [NOAA Lidar Scans](https://coast.noaa.gov/digitalcoast/data/coastallidar.html): public domain
- New annotation for SatlasPretrain: we release these labels under [ODC-BY](https://github.com/allenai/satlas/blob/main/DataLicense).

### Model Weights

We release weights for SatlasNet models pre-trained on SatlasPretrain under [ODC-BY](https://github.com/allenai/satlas/blob/main/DataLicense):

| Image Type | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| Sentinel-2, single-image, RGB | [sentinel2/si_sb_base.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_sb_base.pth) | [sentinel2/si_sb_tiny.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_sb_tiny.pth) | [sentinel2/si_sb_resnet50.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_sb_resnet50.pth) | [sentinel2/si_sb_resnet152.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_sb_resnet152.pth) |
| Sentinel-2, single-image, multi-band | [sentinel2/old_si_mb.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/old_si_mb.pth) | [sentinel2/si_mb_tiny.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_mb_tiny.pth) | [sentinel2/si_mb_resnet50.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_mb_resnet50.pth) | [sentinel2/si_mb_resnet152.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_mb_resnet152.pth) |
| Sentinel-2, multi-image, RGB | [sentinel2/old_mi_sb.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/old_mi_sb.pth) | Unavailable | [sentinel2/mi_sb_resnet50.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/mi_sb_resnet50.pth) | [sentinel2/mi_sb_resnet152.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/mi_sb_resnet152.pth) |
| Sentinel-2, multi-image, multi-band | [sentinel2/old_mi_mb.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/old_mi_mb.pth) | Unavailable | Unavailable | Unavailable |
| NAIP and other high-res, single-image | [highres/old_pretrain.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/highres/old_pretrain.pth) | Unavailable | Unavailable | Unavailable |
| NAIP and other high-res, multi-image | [highres/old_mi.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/highres/old_mi.pth) | Unavailable | Unavailable | Unavailable |
| Landsat 8/9, single-image, multi-band | [landsat/si.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/landsat/si.pth) | Unavailable | Unavailable | Unavailable |
| Landsat 8/9, multi-image, multi-band | [landsat/mi.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/landsat/mi.pth) | Unavailable | Unavailable | Unavailable |
| Sentinel-1, single-image, vh+vv | [sentinel1/si.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel1/si.pth) | Unavailable | Unavailable | Unavailable |
| Sentinel-1, multi-image, vh+vv | [sentinel1/mi.pth](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel1/mi.pth) | Unavailable | Unavailable | Unavailable |

Single-image models learn strong representations for individual satellite or aerial images, while multi-image models use multiple image captures of the same location for added robustness when making predictions about static objects. In multi-image models, feature maps from the backbone are passed through temporal max pooling, so the backbone itself is still applied on individual images, but is trained to provide strong representations after the temporal max pooling step. See [ModelArchitecture.md](ModelArchitecture.md) for more details.

Sentinel-2 RGB models input B2, B3, and B4 only, while the multi-band models input 9 bands (see [Normalization.md](Normalization.md#sentinel-2-images) for details). NAIP models input RGB aerial images, and we have found them to be effective on aerial imagery from a variety of sources and datasets. Landsat models input B1-B11 (all bands).

These documents are useful for using the models:

- [Normalization.md](Normalization.md) documents how images should be normalized for input to the models.
- [See an example of applying the model and visualizing its outputs.](CustomInference.md#sentinel-2-inference-example)
- If you plan to fine-tune the backbone for downstream tasks, [see the example on loading the backbone and extracting feature maps.](CustomInference.md#extracting-representations-example)

The models correspond to configuration files with the same name, e.g. `sentinel2/si_sb_base.pth` is trained using `configs/sentinel2/si_sb_base.txt`. The model weights can be used to compute outputs and stats on the SatlasPretrain validation set, e.g.:

    python -m satlas.cmd.model.infer --config_path configs/sentinel2/si_sb_base.txt --weights models/sentinel2/si_sb_base.pth --details --vis_dir vis/



Dataset Structure
-----------------

SatlasPretrain is divided into high-resolution and low-resolution image modes.
Each image mode has its own train and test split, although dynamic labels have separate files defining how tiles are split than the slow-changing labels.
The splits are defined by JSON files that contain list of (col, row) pairs:

- satlas/metadata/train_lowres.json
- satlas/metadata/test_lowres.json
- satlas/metadata/train_highres.json
- satlas/metadata/test_highres.json
- satlas/metadata/train_event.json
- satlas/metadata/test_event.json


### Tile System

Images and labels are both projected to [Web-Mercator](https://en.wikipedia.org/wiki/Web_Mercator_projection) and stored at zoom level 13.
This means the world is divided into a grid with 2^13 rows and 2^13 columns.
The high-resolution images are stored at zoom level 17.


### Images

For low-resolution image mode, there are:

- Sentinel-2 images in `satlas/sentinel2/`
- Sentinel-1 images in `satlas/sentinel1/`

For high-resolution image mode, there are NAIP images in `satlas/naip/` which are stored at zoom level 17 instead of 13 so that the PNGs are still 512x512.

The image directory structure looks like this:

    satlas/sentinel2/
        S2A_MSIL1C_20160808T181452_N0204_R041_T12SVC_20160808T181447/
            tci/
                1867_3287.png
                1867_3288.png
                ...
            b05/
            b06/
            ...
        S2B_MSIL1C_20221230T180749_N0509_R041_T13UCR_20221230T195817/
            ...
        ...
    satlas/naip/
        m_2508008_nw_17_1_20151013/
            tci/
                36368_55726.png
                ...
            ir/
                36368_55726.png
                ...
        ...


Each folder at the second level contains a different remote sensing image.

The tci, b05, and other sub-folders contain different bands from the image. `tci` contains true-color image; Sentinel-2 includes [other bands](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial) as well.

So `satlas/sentinel2/ABC/tci/1867_3287.png` is a 512x512 image corresponding to column 1867 and row 3287 in the grid, for the tci channel of image `ABC`.
`1867_3288.png` is the image at the next row down.


### Coordinates

The `geo_to_mercator` and `mercator_to_geo` functions in `satlas.util` translate from tile coordinates like (1867, 3287) to longitude-latitude coordinates like (33.4681, -97.9541).

    import satlas.util
    satlas.util.mercator_to_geo((1867, 3287), pixels=1, zoom=13)


### Labels

Satlas consists of slow-changing labels, which should correspond to the most recent image at each tile, and dynamic labels, which reference a specific image/time.

The directory structure looks like this:

    satlas/static/
        1434_3312/
            vector.json
            tree_cover.png
            land_cover.png
        1434_3313/
        ...
    satlas/dynamic/
        1434_3312/
            airplane_325/
                vector.json
            coast_3376/
                vector.json
                water_event.png
            ...
        ...

Each folder at the second level contains labels for a different tile, e.g. `1434_3312` contains labels for the zoom 13 tile at column 1434 and row 3312.

The `airplane_325` and `coast_3376` folders contain dynamic labels for different images. The image name and timestamp is specified under the metadata key in `airplane_325/vector.json`:

    {
        "metadata": {
            "Start": "2014-08-04T23:59:59+00:00",
            "End": "2014-08-05T00:00:01+00:00",
            "ImageUUID": "996e6f0f4f3f42838211caf73c4692f2",
            "ImageName": "m_3211625_sw_11_1_20140805"
        },
        "airplane": [
            ...
        ]
    }

This means these labels correspond to the image at `satlas/naip/m_3211625_sw_11_1_20140805/`.

Point, polygon, polyline, property, and classification labels are stored in `vector.json`. For example:

    satlas/static/1234_5678/vector.json
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
The tasks and categories are defined in `satlas/tasks.py`.

For regression tasks, the value in the greyscale PNG at a pixel is proportional to the quantity (for tree cover, 0 is no trees and 200 is full tree cover; for digital elevation model, 0 is -20 m water depth and 255 is land).

For segmentation tasks, 0 represents invalid pixels, 1 represents the first cateogry in `satlas/metrics/raster.py`, and so on.

For binary segmentation tasks, the rightmost bit in the greyscale value corresponds to the first category, and so on.

Note that only a subset of categories are annotated in each label folder. Oftentimes categories will be annotated but have no instances present in the tile and/or time range, in which case they will appear in `vector.json` like this:

    "power_substation": [],

If the category is not annotated at all, then it will omit the key in `vector.json` entirely (or, for segmentation and regression labels, omit the PNG image like no `land_cover.png`).


### Other Files

Additional files in `satlas/metadata/` contain extra data.

- `train_highres.json`, `train_lowres.json`, `test_event.json`, `test_highres.json`, `train_lowres.json`, `test_event.json` enumerate the tiles assigned to each split.
- `image_times.json` contains the timestamp of each image based on its name.
- `test_highres_property.json` and `test_lowres_property.json` specify the subset of Satlas for evaluating properties. There are also `train_highres_property_1million.json` and `train_lowres_property_1million.json` which contain up to 1 million examples of each property to make the dataset size produced by `satlas.cmd.to_dataset.property`  more tractable, but all of the properties in the train split can be used for training if desired.
- The various `good_images_X.json` files enumerate images that have few cloudy or missing (black) pixels.


### Predicting non-Property Labels

For predicting label types other than properties, the following data from the labels folder can be used:

- Tile position (e.g. `1434_3312`)
- Image name (from `vector.json`)
- The subset of categories annotated in the label folder (must only be used for optimizing inference execution)
- Label folder name (must only be used for creating identically named output folder)


### Predicting Property and Classification Labels

For property and classification labels, the entirety of `vector.json` except the property values can also be used.

Thus, the model can use the coordinates of points, polygons, and polylines like `power_plant` or `road`, and optimize execution based on the property keys (e.g. don't predict road width if it's not labeled for a certain road).

The output should be a new version of `vector.json` with the same features but with the property values filled in based on the model predictions.


### Evaluation

The output format is essentially identical to the format of labels in Satlas. For each label folder like `static/1434_3312/` or `dynamic/1434_3312/airplane_325/`, a corresponding output `outputs/1434_3312/` or `outputs/1434_3312/airplane_325/` should be produced.

    python -m satlas.cmd.evaluate --gt_path path/to/satlas/static/ --pred_path path/to/outputs/ --modality point --split path/to/satlas/metadata/test_highres.json --format static
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/static/ --pred_path path/to/outputs/ --modality polygon --split path/to/satlas/metadata/test_highres.json --format static
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/static/ --pred_path path/to/outputs/ --modality polyline --split path/to/satlas/metadata/test_highres.json --format static
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/static/ --pred_path path/to/outputs/ --modality property --split path/to/satlas/metadata/test_highres.json --format static
    python -m satlas.cmd.evaluate --gt_path path/to/satlas/static/ --pred_path path/to/outputs/ --modality raster --split path/to/satlas/metadata/test_highres.json --format static


Training
---------

### Prepare Datasets

Convert the dataset format to one compatible with training code. It consists of pairs of image time series and subsets of labels under different modalities.

    python -m satlas.cmd.to_dataset.detect --satlas_root satlas_root/ --out_path satlas_root/datasets/
    python -m satlas.cmd.to_dataset.raster --satlas_root satlas_root/ --out_path satlas_root/datasets/
    python -m satlas.cmd.to_dataset.polyline_rasters --dataset_root satlas_root/datasets/
    python -m satlas.cmd.to_dataset.property --satlas_root satlas_root/ --out_path satlas_root/datasets/ --ids satlas_root/metadata/train_lowres_property_100k.json
    python -m satlas.cmd.to_dataset.property --satlas_root satlas_root/ --out_path satlas_root/datasets/ --ids satlas_root/metadata/train_highres_property_100k.json
    python -m satlas.cmd.to_dataset.property --satlas_root satlas_root/ --out_path satlas_root/datasets/ --ids satlas_root/metadata/test_lowres_property.json
    python -m satlas.cmd.to_dataset.property --satlas_root satlas_root/ --out_path satlas_root/datasets/ --ids satlas_root/metadata/test_highres_property.json

Can then visualize these datasets:

    python -m satlas.cmd.vis --path satlas_root/datasets/highres/polygon/ --task polygon --out_path ~/vis/

The format of these datasets is detailed in [DatasetSpec.md](DatasetSpec.md).

### Train Model

Compute weights for each example that balance based on inverse of category frequency:

    python -m satlas.cmd.model.compute_bal_weights --dataset_path satlas_root/datasets/highres/ --out_path satlas_root/bal_weights/highres.json
    python -m satlas.cmd.model.compute_bal_weights --dataset_path satlas_root/datasets/lowres/ --out_path satlas_root/bal_weights/lowres.json

Then train the models:

    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 -m satlas.cmd.model.train --config_path configs/highres_joint.txt --world_size 8
    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 -m satlas.cmd.model.train --config_path configs/lowres_joint.txt --world_size 8

## Infer and Evaluate Model

    python -m satlas.cmd.model.infer --config_path configs/highres_joint.txt --details
    python -m satlas.cmd.model.infer --config_path configs/lowres_joint.txt --details

With visualization:

    python -m satlas.cmd.model.infer --config_path configs/highres_joint.txt --task polygon --details --vis_dir ~/vis/


Fine-tuning
-----------

The code in this repository can also be used to replicate the experiments on downstream datasets. Download the downstream datasets:

    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain_finetune.tar
    tar xvf satlaspretrain_finetune.tar

Example configuration files for 50 training examples are included, e.g.:

    python -m satlas.cmd.model.train --config_path satlaspretrain_finetune/configs/aid_satlas_50.txt
    python -m satlas.cmd.model.train --config_path satlaspretrain_finetune/configs/aid_imagenet_50.txt
    python -m satlas.cmd.model.infer --config_path satlaspretrain_finetune/configs/aid_satlas_50.txt --details
    python -m satlas.cmd.model.infer --config_path satlaspretrain_finetune/configs/aid_imagenet_50.txt --details

The "satlas" configuration files specify `RestorePath` that loads SatlasPretrain weights. `TrainMaxTiles` can be updated for different numbers of training examples.


Authors
-------

- Favyen Bastani
- Piper Wolters
- Ritwik Gupta
- Joe Ferdinando
- Ani Kembhavi

Contact: favyenb@allenai.org or [open an issue](https://github.com/allenai/satlas/issues/new)
