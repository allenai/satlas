[[Satlas Website](https://satlas.allen.ai/) | [Github](https://github.com/allenai/satlas/)]

## Geospatial Data

The AI-generated geospatial data in Satlas can be downloaded using the links below.
All data is released under [ODC-BY](DataLicense).

Visit [satlas.allen.ai](https://satlas.allen.ai/) to view the data in an interactive map.

### Marine Infrastructure (Off-shore Turbines and Platforms)

Marine infrastructure points are available as monthly snapshots in GeoJSON, KML, or shapefile formats.
The latest data can be downloaded at:

- GeoJSON: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.geojson
- KML: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.kml
- Shapefile: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.shp

Replace `latest` with `YYYY-MM` to get an older monthly snapshot. For example, `2023-01.geojson`
contains off-shore wind turbines and platforms that we believe are present as of January 2023.

Each point is annotated with a `category` attribute, either `offshore_wind_turbine` or `offshore_platform`.

An index file lists all currently available files, along with their hash and last modified time.
You can programatically query this file to detect updates, which generally will be published
between the 15th and 20th of each month.
Data for previous months will be updated when (a) smoothing that includes the latest images
changes our estimates for previous months; or (b) we develop improved models.

- Index: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/index.txt

The points are also available as a single GeoJSON, with additional `start` and `end` attributes
indicating when the object was constructed or removed (e.g. `start=2021-05,end=2022-01`):

- GeoJSON: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/marine.geojson

### Tree Cover

Tree cover is available as GeoTIFF files for each WebMercator tile at zoom 7.
WebMercator tiles are indexed from the top-left of the projection coordinate
system from (0, 0) to (127, 127). The resolution is 10 m/pixel.

Replace YYYY-MM with the padded year and month (e.g. 2023-02 or 2023-11), along with the
unpadded X and Y index of the tile (like `20_44.tif` for Seattle).
Currently the data is computed for 2016-01 to 2023-01 only in 01 (January) and 07 (July).

https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/tree-cover/YYYY-MM/X_Y.tif

For example, for Seattle in 2022-07:

https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/tree-cover/2022-07/20_44.tif

The GeoTIFF contains a single 8-bit band, with these values:

    0: no data
    1: no trees
    2: low
    3: medium
    4: high
    5: full tree cover

### Renewable Energy Infrastructure (On-shore Wind Turbines and Solar Farms)

Renewable energy infrastructure points (on-shore wind turbines) and polygons (solar farms) are available as monthly snapshots in GeoJSON, KML, or shapefile formats.
The latest data can be downloaded at:

- GeoJSON: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/latest.geojson
- KML: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/latest.kml
- Shapefile (wind turbines): https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/latest_wind.shp
- Shapefile (solar farms): https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/latest_solar.shp

Replace `latest` with `YYYY-MM` to get an older monthly snapshot. For example, `2023-01.geojson`
contains wind turbines and solar farms that we believe are present as of January 2023.

Each point is annotated with a `category` attribute, either `wind_turbine` or `solar_farm`.

An index file lists all currently available files, along with their hash and last modified time.
You can programatically query this file to detect updates, which generally will be published
between the 15th and 20th of each month.
Data for previous months will be updated when (a) smoothing that includes the latest images
changes our estimates for previous months; or (b) we develop improved models.

- Index: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/index.txt

The data is also available as a single GeoJSON, with additional `start` and `end` attributes
indicating when the object was constructed or removed (e.g. `start=2021-05,end=2022-01`).
Solar farms in this combined format are not as accurate, since they are split up when different
parts of the solar farm were built at different times.

- GeoJSON: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/renewable.geojson

### Super-Resolution

Due to the large size, the super-resolved imagery is not currently available for bulk download.

For the training and inference code and pre-trained model weights, see https://github.com/allenai/satlas-super-resolution/.

## Contact

If you have feedback about the code, data, or models, or if you would like to see new types of geospatial data that are feasible to produce from Sentinel-2 imagery,
you can contact us by [opening an issue](https://github.com/allenai/satlas/issues/new) or via e-mail at satlas@allenai.org.