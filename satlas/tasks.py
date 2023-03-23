# Point and polygon tasks.
point_categories = [
    'wind_turbine', 'lighthouse', 'mineshaft', 'aerialway_pylon', 'helipad',
    'fountain', 'toll_booth', 'chimney', 'communications_tower',
    'flagpole', 'petroleum_well', 'water_tower',
    'offshore_wind_turbine', 'offshore_platform',
    'power_tower',
]

polygon_categories = [
    'aquafarm', 'lock', 'dam', 'solar_farm', 'power_plant', 'gas_station',
    'park', 'parking_garage', 'parking_lot', 'landfill', 'quarry', 'stadium',
    'airport', 'airport_apron', 'airport_hangar', 'airport_terminal',
    'ski_resort', 'theme_park', 'storage_tank', 'silo', 'track',
    'wastewater_plant', 'power_substation', 'pier', 'crop',
]

polyline_categories = [
    'airport_runway', 'airport_taxiway', 'raceway', 'road', 'railway', 'river',
]

detect_tasks = [{
    'name': 'point',
    'type': 'point',
    'categories': point_categories,
    'image_type': 'all',
}, {
    'name': 'polygon',
    'type': 'polygon',
    'categories': polygon_categories,
    'image_type': 'all',
}, {
    'name': 'airplane',
    'type': 'point',
    'categories': ['airplane'],
    'image_type': 'highres',
}, {
    'name': 'rooftop_solar_panel',
    'type': 'point',
    'categories': ['rooftop_solar_panel'],
    'image_type': 'highres',
}, {
    'name': 'building',
    'type': 'polygon',
    'categories': ['ms_building'],
    'image_type': 'highres',
}, {
    'name': 'vessel',
    'type': 'point',
    'categories': ['vessel'],
    'image_type': 'lowres',
}]

polyline_tasks = [{
    'name': 'polyline',
    'type': 'polyline',
    'categories': polyline_categories,
    'image_type': 'all',
}]

# Raster tasks.
raster_tasks = [{
    'name': 'land_cover',
    'id': 'land_cover',
    'type': 'segment',
    'categories': ['invalid', 'water', 'developed', 'tree', 'shrub', 'grass', 'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss'],
    'image_type': 'all',
    'colors': [
        [0, 0, 0], # (black) invalid
        [0, 0, 255], # (blue) water
        [255, 0, 0], # (red) developed
        [0, 192, 0], # (dark green) tree
        [200, 170, 120], # (brown) shrub
        [0, 255, 0], # (green) grass
        [255, 255, 0], # (yellow) crop
        [128, 128, 128], # (grey) bare
        [255, 255, 255], # (white) snow
        [0, 255, 255], # (cyan) wetland
        [255, 0, 255], # (pink) mangroves
        [128, 0, 128], # (purple) moss
    ],
}, {
    'name': 'dem',
    'id': 'dem',
    'type': 'regress',
    'categories': None,
    'image_type': 'all',
}, {
    'name': 'crop_type',
    'id': 'crop_type',
    'type': 'segment',
    'categories': ['invalid', 'rice', 'grape', 'corn', 'sugarcane', 'tea', 'hop', 'wheat', 'soy', 'barley', 'oats', 'rye', 'cassava', 'potato', 'sunflower', 'asparagus', 'coffee'],
    'image_type': 'all',
}, {
    'name': 'tree_cover',
    'id': 'tree_cover',
    'type': 'regress',
    'categories': None,
    'image_type': 'all',
}, {
    'name': 'water_event',
    'id': 'water_event',
    'type': 'segment',
    'categories': ['invalid', 'background', 'water_event'],
    'image_type': 'highres',
    'colors': [
        [0, 0, 0], # (black) invalid
        [0, 255, 0], # (green) background
        [0, 0, 255], # (blue) water_event
    ],
}, {
    'name': 'flood',
    'id': 'water',
    'type': 'segment',
    'categories': ['invalid', 'background', 'flood'],
    'image_type': 'lowres',
}, {
    'name': 'cloud',
    'id': 'cloud',
    'type': 'segment',
    'categories': ['invalid', 'background', 'cloud', 'shadow'],
    'image_type': 'lowres',
}, {
    'name': 'wildfire',
    'id': 'wildfire',
    'type': 'bin_segment',
    'categories': ['fire_retardant', 'burned'],
    'image_type': 'lowres',
    'colors': [
        [255, 0, 0], # (red) fire retardant
        [128, 128, 128], # (grey) burned area
    ]
}, {
    'name': 'polyline_bin_segment',
    'type': 'bin_segment',
    'categories': polyline_categories,
    'image_type': 'all',
    'colors': [
        [255, 255, 255], # (white) airport_runway
        [192, 192, 192], # (light grey) airport_taxiway
        [160, 82, 45], # (sienna) raceway
        [255, 255, 255], # (white) road
        [144, 238, 144], # (light green) railway
        [0, 0, 255], # (blue) river
    ]
}]

# Property and classification tasks.
property_tasks = [{
    'name': 'wind_turbine_rotor_diameter',
    'obj': 'wind_turbine',
    'property': 'rotor_diameter',
    'type': 'property_numeric',
    'image_type': 'all',
}, {
    'name': 'wind_turbine_power_mw',
    'obj': 'wind_turbine',
    'property': 'power_mw',
    'type': 'property_numeric',
    'image_type': 'all',
}, {
    'name': 'parking_lot_capacity',
    'obj': 'parking_lot',
    'property': 'capacity',
    'type': 'property_numeric',
    'image_type': 'all',
}, {
    'name': 'track_sport',
    'obj': 'track',
    'property': 'sport',
    'type': 'property_category',
    'categories': ['running', 'cycling', 'horse'],
    'image_type': 'all',
}, {
    'name': 'road_type',
    'obj': 'road',
    'property': 'road_type',
    'type': 'property_category',
    'categories': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'track', 'pedestrian'],
    'image_type': 'all',
}, {
    'name': 'road_lanes',
    'obj': 'road',
    'property': 'lanes',
    'type': 'property_numeric',
    'image_type': 'all',
}, {
    'name': 'road_speed',
    'obj': 'road',
    'property': 'max_speed',
    'type': 'property_numeric',
    'image_type': 'all',
}, {
    'name': 'road_bridge',
    'obj': 'road',
    'property': 'bridge',
    'type': 'property_numeric',
    'image_type': 'all',
}, {
    'name': 'power_plant_type',
    'obj': 'power_plant',
    'property': 'plant_type',
    'type': 'property_category',
    'categories': ['oil', 'nuclear', 'coal', 'gas'],
    'image_type': 'all',
}, {
    'name': 'quarry_resource',
    'obj': 'quarry',
    'property': 'resource',
    'type': 'property_category',
    'categories': ['sand', 'gravel', 'clay', 'coal', 'peat'],
    'image_type': 'all',
}, {
    'name': 'park_type',
    'obj': 'park',
    'property': 'park_type',
    'type': 'property_category',
    'categories': ['park', 'pitch', 'golf_course', 'cemetery'],
    'image_type': 'all',
}, {
    'name': 'park_sport',
    'obj': 'park',
    'property': 'sport',
    'type': 'property_category',
    'categories': ['american_football', 'badminton', 'baseball', 'basketball', 'cricket', 'rugby', 'soccer', 'tennis', 'volleyball'],
    'image_type': 'all',
}, {
    'name': 'smoke',
    'obj': 'smoke',
    'property': 'smoke',
    'type': 'classify',
    'categories': ['no', 'partial', 'yes'],
    'image_type': 'lowres',
}, {
    'name': 'snow',
    'obj': 'snow',
    'property': 'snow',
    'type': 'classify',
    'categories': ['no', 'partial', 'yes'],
    'image_type': 'lowres',
}]
