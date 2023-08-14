Dataset Spec
------------

Below we document the format of datasets for use with this codebase.
Note that [SatlasPretrain](SatlasPretrain.md) uses a different format, but the scripts in `satlas.cmd.to_dataset` convert SatlasPretrain to a set of datasets under the format below.

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

Each folder like `4267_2839/` contains a different training example, which corresponds to a particular geographic location.
The `images/` subfolder contains images captured at different times at that location.

Besides the SatlasPretrain high-res datasets, the current datasets use [Sentinel-2 images](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). `tci.png` contains B02, B03, and B04. The 10 m/pixel and 20 m/pixel bands (except B8A) are used as input and included in the training data, while the 60 m/pixel bands are not used.

The Sentinel-2 bands were normalized to 8-bit PNGs as follows (see [Normalization.md](Normalization.md) for more details, note that the 8-bit values are divided by 255 when passing to the model):
- `tci`: taken from the TCI JPEG2000 image provided by ESA. This is already an 8-bit RGB product.
- Other bands: the raw image scenes from ESA are 16-bit products. We convert to greyscale 8-bit PNGs: `clip(band/32, 0, 255)`.

For segmentation (solar farm) and regression (tree cover) labels, `gt.png` contains a greyscale mask. For segmentation, the pixel value indicates the class ID. For regression, the pixel value indicates the ground truth value.

For object detection labels (on-shore wind turbines and marine infrastructure), `gt.json` contains bounding box labels like this:

    [
        [14, 467, 54, 507, "wind_turbine"],
        [53, 473, 93, 513, "wind_turbine"]
    ]

Each box is in the form `[start_col, start_row, end_col, end_row, category_name]`.
The current fine-tuning tasks (i.e., non-SatlasPretrain datasets) are annotated as points so the boxes are all the same size, with the center point being the actual label. The center point can be easily derived `[(start_col + end_col) / 2, (start_row + end_row) / 2]` if desired.
