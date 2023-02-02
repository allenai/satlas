Download URL: https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-marine-infrastructure-v1.zip

The marine infrastructure dataset contains labels for three classes:

- Vessels
- Off-shore Wind Turbines
- Off-shore Platforms

The directory structure looks like:

    1044_2500_cce5490ca2dc4208bb71acf23122c387/
        gt.json
        images/
            cce5490ca2dc4208bb71acf23122c387/
                tci.png
                virtual:overlap_tci_0.png
                virtual:overlap_tci_1.png
                virtual:overlap_tci_2.png
                virtual:overlap_tci_3.png
    1068_1872_a0e69f83b8354016bc020c8878649930/
    ...

The PNGs contain 1024x1024 windows of spatially aligned Sentinel-2 images from different timestamps.
The `tci.png`, `_0`, `_1`, `_2`, and `_3` images should be in descending temporal order.

Vessels are labeled in `tci.png`.
Labeling of wind turbines and platforms is based primarily on the first three images.

The labels are in `gt.json` with format like this:

    [
        [991, 190, 1031, 230, "vessel"],
        [822, 648, 862, 688, "turbine"],
        [993, 457, 1033, 497, "platform"]
    ]

This specifies three bounding boxes, with `[x1, y1, x2, y2, class_label]` for each one.

The labels are actually points, so all boxes are the same size centered at these points.
You can use the boxes, or you can compute `(x1+x2)/2` and `(y1+y2)/2` and use the original points directly.

The marine infrastructure dataset is released under Apache License 2.0.