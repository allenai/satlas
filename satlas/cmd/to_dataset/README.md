This code converts Satlas to a set of per-task datasets that consist of pairs of image time series and labels.

Each produced dataset corresponds to a subset of categories.

First run get_good_images, this creates files in metadata folder that exclude
images with many missing or cloud/snow pixels:

    python -m satlas.cmd.to_dataset.get_good_images --satlas_root /satlas/

Then:

    python -m satlas.cmd.to_dataset.detect --satlas_root /satlas/ --split test_highres --out_path /satlas/datasets/
    python -m satlas.cmd.to_dataset.raster --satlas_root /satlas/ --split test_highres --out_path /satlas/datasets/
    python -m satlas.cmd.to_dataset.property --satlas_root /satlas/ --split test_highres --out_path /satlas/datasets/

And can visualize it:

    python -m satlas.cmd.vis --path /satlas/datasets/highres_polygon/ --task polygon --out_path ~/vis/

Limitations:
- Only includes "tci" channel.
- Limited spatial extent.