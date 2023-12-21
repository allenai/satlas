This document describes how band values are normalized for input into Satlas models.

All bands are normalized to 0-1.


High-Resolution Images
----------------------

For NAIP: remove the IR channel if any, and divide the 0-255 RGB values by 255.

For other RGB high-resolution images, e.g. Google Earth, divide the 0-255 RGB values by 255.

Below is an example of downloading an image and preparing it for input into a Satlas model.

### High-Resolution Example

First find the longitude and latitude of the location you're interested in, and convert it to a Web-Mercator tile at zoom 16-18. You can use the [Satlas Map](https://satlas.allen.ai/map) and hover your mouse over a point of interest to get its longitude and latitude. To convert to tile using Python:

```python
import satlas.util
longitude = -122.333
latitude = 47.646
print(satlas.util.geo_to_mercator((longitude, latitude), pixels=1, zoom=17))
```

Get a high-resolution image that you want to apply the model on, e.g. you could download an image from Google Maps by visiting a URL like this:

    http://mt1.google.com/vt?lyrs=s&x={x}&y={y}&z={z}
    Example: http://mt1.google.com/vt?lyrs=s&x=20995&y=45754&z=17

Let's assume the image is saved as `image.jpg`. Load and normalize the image as follows:

```python
import torchvision
im = torchvision.io.read_image(image_path)
im = im.float() / 255
# Now you can apply model on the image:
# outputs, _ = model([im])
```


Sentinel-2 Images
-----------------

3-band Satlas Sentinel-2 models input the TCI file provided with the scene only.
This contains an 8-bit image that has been normalized by ESA to the 0-255 range.
The image is normalized for input to the model by dividing the 0-255 RGB values by 255, and retaining the RGB order.

The multi-band Sentinel-2 models input 9 channels, with the non-TCI channels normalized from the 16-bit source data by dividing by 8160 and clipping to 0-1. The channels are ordered as follows:
- TCI (this is three channels, RGB; divide by 255)
- B05, B06, B07, B08, B11, B12 (divide by 8160, clip to 0-1)

This order is specified in the config file e.g. `configs/satlas_explorer_solar_farm.txt`.

Below is an example of downloading three Sentinel-2 images, normalizing the bands, and saving it as a `.npy` which can then be read and run through a multi-image, multi-band Satlas model:

### Sentinel-2 Example

First use [scihub.copernicus.eu](https://scihub.copernicus.eu/dhus/) to download three Sentinel-2 scenes of the same location.

1. Create an account using the profile icon in the top-right.
2. Zoom in on a location of interest, and use the rectangle tool (middle right, the square with dotted lines icon) to draw a rectangle.
3. Open the filters (three horizontal bars icon) in the top-left, check "Mission: Sentinel-2", and select S2MSI1C for product type. Optionally limit cloud cover to "[0 TO 20]" or similar. Optionally add start/end times under Sensing Period.
4. Press the search button. You should see a list of Sentinel-2 scenes, and when you hover over one of them it should highlight the scene on the map.
5. Find three scenes covering the same geographic extent (based on what's highlighted in the map when you hover over that item in the product list) and download them.
6. Unzip the files into `scenes/`.

Use gdal to merge the bands across scenes. We include nine bands since we're assuming use of a multi-band model. If you're using a TCI-only model, then just read the TCI band below.

```python
import glob
import os
import subprocess
channels = ['B08', 'TCI', 'B05', 'B06', 'B07', 'B11', 'B12']
fnames = []
for scene_name in os.listdir('scenes'):
    image_fnames = glob.glob(os.path.join('scenes/{}/GRANULE/L1C_*/IMG_DATA/*.jp2'.format(scene_name)))
    channel_to_fname = {fname.split('_')[-1].split('.')[0]: fname for fname in image_fnames}
    selected_fnames = [channel_to_fname[channel] for channel in channels]
    fnames.extend(selected_fnames)
subprocess.call([
    'gdal_merge.py',
    '-o', 'stack.tif',
    # Keep bands separate in output file.
    '-separate',
] + fnames)
```

The model expects bands in a different order with TCI first, but we put B08 first so that gdal_merge creates a 10 m/pixel 16-bit output image.

Now load the images, normalize them, and save the result:

```python
import numpy as np
from osgeo import gdal
raster = gdal.Open('stack.tif')
image = raster.ReadAsArray()
# Separate out the different 9-band images.
image = image.reshape(3, 9, image.shape[1], image.shape[2])
# Re-order bands to the order expected by the model.
image = image[:, (1, 2, 3, 4, 5, 6, 0, 7, 8), :, :]
# Normalize the non-TCI bands to be 0-255.
image[:, 3:9, :, :] = np.clip(image[:, 3:9, :, :]//32, 0, 255)
# Save the 8-bit image.
image = image.astype(np.uint8)
np.save('stack.npy', image)
```


Landsat Images
--------------

Our Landsat models input 11 bands, B1-B11 in order, of Landsat-8 and Landsat-9 images.

Each band is originally a 16-bit image. We normalize a pixel value N to 0-1 by clipping (N-4000)/16320 to 0-1.


Sentinel-1 Images
-----------------

Our Sentinel-1 models input 2 bands, vh then vv, of Level-1 GRD IW vh+vv products only (10 m pixel spacing).

Each band is originally a 16-bit image. We normalize a pixel value N to 0-1 by clipping N/255 to 0-1 (any pixel values greater than 255 become 1).
