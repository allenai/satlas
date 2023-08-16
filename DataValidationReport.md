## Data Validation Report

For each Satlas geospatial data product, we estimated the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
of the data as of 2022-10 on a per-continent basis.

Precision is the percentage of objects in the geospatial data products that are correct (not errors),
while recall is the percentage of actual objects in the world that are covered in the geospatial data (not missing).

We estimated precision by uniformly sampling 100 objects from the 2022-10 data within each continent, and manually determining whether each sampled object was correct.
Sometimes, a confident determination could not be made due to the lack of sufficient data, in which case we made a best effort guess.
The precision percentages reported below are equal to the number of objects we determined were correct in each continent.
Missing values indicate that fewer than 100 objects were found in that continent.

We estimated recall using our hand-labeled validation sets.
Each recall percentage is the percentage of ground truth objects in the validation set within each continent that was output by our model, at the same threshold we used for deployment.
Missing values indicate that there were insufficient ground truth objects in that continent.

"Objects" are points for wind turbines and platforms. for solar farms, objects are predicted polygons for precision, and individual pixels for recall.

Below, Count shows the number of objects in the geospatial data products.

This report may change as the data is improved over time. The current version is last modified 16 August 2023.

## Solar Farms

| Continent     | Precision | Recall | Count   |
| ---------     | --------- | ------ | -----   |
| Africa        |        57 |    N/A |   1,051 |
| Asia          |        97 |   77.9 |  87,197 |
| Europe        |        94 |   78.2 |  42,471 |
| North America |        86 |   89.1 |  15,493 |
| South America |        76 |   67.8 |   2,091 |
| Oceania       |        74 |   57.4 |     797 |
| All           |        95 |   78.9 | 149,100 |

Many incorrect solar farms were actually greenhouses, but other errors were diverse, often occurring on undeveloped land that was dark or metallic in appearance.
Errors occur roughly proportionally to land area, so regions with the most actual solar farms (Asia, Europe, North America) have the highest precision.

## Onshore Wind Turbines

| Continent     | Precision | Recall | Count   |
| ---------     | --------- | ------ | -----   |
| Africa        |       100 |    N/A |   2,223 |
| Asia          |       100 |   96.7 | 130,233 |
| Europe        |        98 |   92.5 |  78,283 |
| North America |       100 |   87.5 |  69,161 |
| South America |        99 |    N/A |   8,371 |
| Oceania       |        94 |    N/A |   3,471 |
| All           |        99 |   93.2 | 291,742 |

Most (6/9) incorrect onshore wind turbines were actually transmission towers. Other incorrect turbines were rare mistakes in empty sites (1/9) or undeveloped land (2/9).

## Offshore Wind Turbines

| Continent     | Precision | Recall | Count  |
| ---------     | --------- | ------ | -----  |
| Africa        |       N/A |    N/A |      0 |
| Asia          |        99 |   91.1 |  5,509 |
| Europe        |       100 |   97.0 |  5,753 |
| North America |       N/A |    N/A |     10 |
| South America |       N/A |    N/A |      0 |
| Oceania       |       N/A |    N/A |      0 |
| All           |        99 |   94.1 | 11,272 |

Only one offshore incorrect wind turbine was identified, in Asia, which was actually a transmission tower in the water at (25.2260, 119.1385).

## Offshore Platforms

| Continent     | Precision | Recall | Count  |
| ---------     | --------- | ------ | -----  |
| Africa        |        90 |   90.6 |    842 |
| Asia          |        96 |   48.3 |  6,531 |
| Europe        |        88 |   66.1 |  1,087 |
| North America |        94 |   87.5 |  2,277 |
| South America |        96 |    N/A |  1,329 |
| Oceania       |        81 |    N/A |    126 |
| All           |        94 |   61.7 | 12,192 |

Incorrect offshore platforms were generally islands or locations frequently occupied by anchored fishing vessels.

## Tree Cover

Data validation is currently unavailable for tree cover.
