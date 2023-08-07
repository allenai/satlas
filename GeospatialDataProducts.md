[[Website](https://satlas.allen.ai/) | [Github](https://github.com/allenai/satlas/)]

## Geospatial Data

The AI-generated geospatial data in Satlas can be downloaded using the links below.
All data is released under [ODC-BY](DataLicense).

Visit [satlas.allen.ai](https://satlas.allen.ai/) to view the data in an interactive map.

### Marine Infrastructure (Off-shore Turbines and Platforms)

Marine infrastructure points are available as a single GeoJSON or KML file:

- GeoJSON: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/marine.geojson
- KML: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/marine.kml

Each point is annotated with these attributes:
- `category`: either `offshore_wind_turbine` or `offshore_platform`
- `start`: when the object was constructed
- `end`: when the object was removed, or `2022-12` if still present

Points are also available as snapshots at different points in time, in which case
the file only contains points present at that time (rather than the `start` and
`end` attributes.) Replace YYYY-MM with a year between 2016 and 2022 and month
between 01 and 12:

https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/YYYY-MM.geojson

### Tree Cover

Tree cover is available as GeoTIFF files for each WebMercator tile at zoom 7.
WebMercator tiles are indexed from the top-left of the projection coordinate
system from (0, 0) to (127, 127). The resolution is 10 m/pixel.

Replace YYYY-MM with a year between 2016 and 2022 and month either 01 or 07,
along with the unpadded X and Y index of the tile (like 20_44.tif for Seattle):

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

Renewable energy infrastructure points (on-shore wind turbines) and polygons (solar farms) are available as monthly GeoJSON files:

- GeoJSON: https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/renewable/YYYY-MM.geojson

Replace YYYY-MM with a year between 2016 and 2022, and month either 01 or 07.

Each feature is annotated with these attributes:
- `category`: either `wind_turbine` or `solar_farm`

### Super-Resolution

Due to the large size, the super-resolved imagery is not currently available for bulk download.

For the training and inference code and pre-trained model weights, see https://github.com/allenai/satlas-super-resolution/.
