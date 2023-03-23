import hashlib
import zlib
import json
import numpy as np
import os
import random
import skimage.io
import skimage.draw
import torch
import torchvision

import satlas.util

chip_size = 512

tasks = {
    'polyline_bin_segment': {
        'type': 'bin_segment',
        'categories': [
            'airport_runway', 'airport_taxiway', 'raceway', 'road', 'railway', 'river',
        ],
        'colors': [
            [255, 255, 255], # (white) airport_runway
            [192, 192, 192], # (light grey) airport_taxiway
            [160, 82, 45], # (sienna) raceway
            [255, 255, 255], # (white) road
            [144, 238, 144], # (light green) railway
            [0, 0, 255], # (blue) river
        ],
    },
    'bin_segment': {
        'type': 'bin_segment',
        'categories': [
            "aquafarm", "lock", "dam", "solar_farm", "power_plant", "gas_station",
            "park", "parking_garage", "parking_lot", "landfill", "quarry",
            "stadium", "airport", "airport_runway", "airport_taxiway",
            "airport_apron", "airport_hangar", "airstrip", "airport_terminal",
            "ski_resort", "theme_park", "storage_tank", "silo", "track",
            "raceway", "wastewater_plant", "road", "railway", "river",
            "water_park", "pier", "water_tower", "street_lamp", "traffic_signals",
            "power_tower", "power_substation", "building", "bridge",
            "road_motorway", "road_trunk", "road_primary", "road_secondary", "road_tertiary",
            "road_residential", "road_service", "road_track", "road_pedestrian",
        ],
        'colors': [
            [32, 178, 170], # (light sea green) aquafarm
            [0, 255, 255], # (cyan) lock
            [173, 216, 230], # (light blue) dam
            [255, 0, 255], # (magenta) solar farm
            [255, 165, 0], # (orange) power plant
            [128, 128, 0], # (olive) gas station
            [0, 255, 0], # (green) park
            [47, 79, 79], # (dark slate gray) parking garage
            [128, 0, 0], # (maroon) parking lot
            [165, 42, 42], # (brown) landfill
            [128, 128, 128], # (grey) quarry
            [255, 215, 0], # (gold) stadium
            [255, 105, 180], # (pink) airport
            [255, 255, 255], # (white) airport_runway
            [192, 192, 192], # (light grey) airport_taxiway
            [128, 0, 128], # (purple) airport_apron
            [0, 128, 0], # (dark green) airport_hangar
            [248, 248, 255], # (ghost white) airstrip
            [240, 230, 140], # (khaki) airport_terminal
            [192, 192, 192], # (silver) ski_resort
            [0, 96, 0], # (dark green) theme_park
            [95, 158, 160], # (cadet blue) storage_tank
            [205, 133, 63], # (peru) silo
            [154, 205, 50], # (yellow green) track
            [160, 82, 45], # (sienna) raceway
            [218, 112, 214], # (orchid) wastewater_plant
            [255, 255, 255], # (white) road
            [144, 238, 144], # (light green) railway
            [0, 0, 255], # (blue) river
            [255, 240, 245], # (lavender blush) water_park
            [65, 105, 225], # (royal blue) pier
            [238, 130, 238], # (violet) water_tower
            [75, 0, 130], # (indigo) street_lamp
            [233, 150, 122], # (dark salmon) traffic_signals
            [255, 255, 0], # (yellow) power_tower
            [255, 255, 0], # (yellow) power_substation
            [255, 0, 0], # (red) building
            [64, 64, 64], # (dark grey) bridge
            [255, 255, 255], # (white) road_motorway
            [255, 255, 255], # (white) road_trunk
            [255, 255, 255], # (white) road_primary
            [255, 255, 255], # (white) road_secondary
            [255, 255, 255], # (white) road_tertiary
            [255, 255, 255], # (white) road_residential
            [255, 255, 255], # (white) road_service
            [255, 255, 255], # (white) road_track
            [255, 255, 255], # (white) road_pedestrian
        ],
    },
    'land_cover': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'background',
            'water', 'developed', 'tree', 'shrub', 'grass',
            'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss',
        ],
        'colors': [
            [0, 0, 0], # unknown
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
    },
    'tree_cover': {
        'type': 'regress',
        'BackgroundInvalid': True,
    },
    'crop_type': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'invalid',
            'rice', 'grape', 'corn', 'sugarcane',
            'tea', 'hop', 'wheat', 'soy', 'barley',
            'oats', 'rye', 'cassava', 'potato', 'sunflower', 'asparagus', 'coffee',
        ],
        'colors': [
            [0, 0, 0], # unknown
            [0, 0, 255], # (blue) rice
            [255, 0, 0], # (red) grape
            [255, 255, 0], # (yellow) corn
            [0, 255, 0], # (green) sugarcane
            [128, 0, 128], # (purple) tea
            [255, 0, 255], # (pink) hop
            [0, 128, 0], # (dark green) wheat
            [255, 255, 255], # (white) soy
            [128, 128, 128], # (grey) barley
            [165, 42, 42], # (brown) oats
            [0, 255, 255], # (cyan) rye
            [128, 0, 0], # (maroon) cassava
            [173, 216, 230], # (light blue) potato
            [128, 128, 0], # (olive) sunflower
            [0, 128, 0], # (dark green) asparagus
            [92, 64, 51], # (dark brown) coffee
        ],
    },
    'point': {
        'type': 'detect',
        'categories': [
            'background',
            'wind_turbine', 'lighthouse', 'mineshaft', 'aerialway_pylon', 'helipad',
            'fountain', 'toll_booth', 'chimney', 'communications_tower',
            'flagpole', 'petroleum_well', 'water_tower',
            'offshore_wind_turbine', 'offshore_platform', 'power_tower',
        ],
        'colors': [
            [0, 0, 0],
            [0, 255, 255], # (cyan) wind_turbine
            [0, 255, 0], # (green) lighthouse
            [255, 255, 0], # (yellow) mineshaft
            [0, 0, 255], # (blue) pylon
            [173, 216, 230], # (light blue) helipad
            [128, 0, 128], # (purple) fountain
            [255, 255, 255], # (white) toll_booth
            [0, 128, 0], # (dark green) chimney
            [128, 128, 128], # (grey) communications_tower
            [165, 42, 42], # (brown) flagpole
            [128, 0, 0], # (maroon) petroleum_well
            [255, 165, 0], # (orange) water_tower
            [255, 255, 0], # (yellow) offshore_wind_turbine
            [255, 0, 0], # (red) offshore_platform
            [255, 0, 255], # (magenta) power_tower
        ],
    },
    'rooftop_solar_panel': {
        'type': 'detect',
        'categories': [
            'background',
            'rooftop_solar_panel',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0], # (yellow) rooftop_solar_panel
        ],
    },
    'building': {
        'type': 'instance',
        'categories': [
            'background',
            'ms_building',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0], # (yellow) building
        ],
    },
    'polygon': {
        'type': 'instance',
        'categories': [
            'background',
            'aquafarm', 'lock', 'dam', 'solar_farm', 'power_plant', 'gas_station',
            'park', 'parking_garage', 'parking_lot', 'landfill', 'quarry', 'stadium',
            'airport', 'airport_apron', 'airport_hangar', 'airport_terminal',
            'ski_resort', 'theme_park', 'storage_tank', 'silo', 'track',
            'wastewater_plant', 'power_substation', 'pier', 'crop',
            'water_park',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0], # (yellow) aquafarm
            [0, 255, 255], # (cyan) lock
            [0, 255, 0], # (green) dam
            [0, 0, 255], # (blue) solar_farm
            [255, 0, 0], # (red) power_plant
            [128, 0, 128], # (purple) gas_station
            [255, 255, 255], # (white) park
            [0, 128, 0], # (dark green) parking_garage
            [128, 128, 128], # (grey) parking_lot
            [165, 42, 42], # (brown) landfill
            [128, 0, 0], # (maroon) quarry
            [255, 165, 0], # (orange) stadium
            [255, 105, 180], # (pink) airport
            [192, 192, 192], # (silver) airport_apron
            [173, 216, 230], # (light blue) airport_hangar
            [32, 178, 170], # (light sea green) airport_terminal
            [255, 0, 255], # (magenta) ski_resort
            [128, 128, 0], # (olive) theme_park
            [47, 79, 79], # (dark slate gray) storage_tank
            [255, 215, 0], # (gold) silo
            [192, 192, 192], # (light grey) track
            [240, 230, 140], # (khaki) wastewater_plant
            [154, 205, 50], # (yellow green) power_substation
            [255, 165, 0], # (orange) pier
            [0, 192, 0], # (middle green) crop
            [0, 192, 0], # (middle green) water_park
        ],
    },
    'wildfire': {
        'type': 'bin_segment',
        'categories': ['fire_retardant', 'burned'],
        'colors': [
            [255, 0, 0], # (red) fire retardant
            [128, 128, 128], # (grey) burned area
        ],
    },
    'smoke': {
        'type': 'classification',
        'categories': ['no', 'partial', 'yes'],
    },
    'snow': {
        'type': 'classification',
        'categories': ['no', 'partial', 'yes'],
    },
    'dem': {
        'type': 'regress',
        'BackgroundInvalid': True,
    },
    'airplane': {
        'type': 'detect',
        'categories': ['background', 'airplane'],
        'colors': [
            [0, 0, 0], # (black) background
            [255, 0, 0], # (red) airplane
        ],
    },
    'vessel': {
        'type': 'detect',
        'categories': ['background', 'vessel'],
        'colors': [
            [0, 0, 0], # (black) background
            [255, 0, 0], # (red) vessel
        ],
    },
    'water_event': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': ['invalid', 'background', 'water_event'],
        'colors': [
            [0, 0, 0], # (black) invalid
            [0, 255, 0], # (green) background
            [0, 0, 255], # (blue) water_event
        ],
    },
    'park_sport': {
        'type': 'classification',
        'categories': ['american_football', 'badminton', 'baseball', 'basketball', 'cricket', 'rugby', 'soccer', 'tennis', 'volleyball'],
    },
    'park_type': {
        'type': 'classification',
        'categories': ['park', 'pitch', 'golf_course', 'cemetery'],
    },
    'power_plant_type': {
        'type': 'classification',
        'categories': ['oil', 'nuclear', 'coal', 'gas'],
    },
    'quarry_resource': {
        'type': 'classification',
        'categories': ['sand', 'gravel', 'clay', 'coal', 'peat'],
    },
    'track_sport': {
        'type': 'classification',
        'categories': ['running', 'cycling', 'horse'],
    },
    'road_type': {
        'type': 'classification',
        'categories': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'track', 'pedestrian'],
    },
    'cloud': {
        'type': 'bin_segment',
        'categories': ['background', 'cloud', 'shadow'],
        'colors': [
            [0, 255, 0], # (green) not clouds or shadows
            [255, 255, 255], # (white) clouds
            [128, 128, 128], # (grey) shadows
	    ],
        'BackgroundInvalid': True,
    },
    'flood': {
        'type': 'bin_segment',
        'categories': ['background', 'water'],
        'colors': [
            [0, 255, 0], # (green) background
            [0, 0, 255], # (blue) water
        ],
        'BackgroundInvalid': True,
    },
}

def get_invalid_target(task):
    task_type = task['type']

    if task_type == 'segment' or task_type == 'regress':
        return {
            'valid': torch.tensor(0, dtype=torch.int32),
            'valid_im': torch.zeros((chip_size, chip_size), dtype=torch.bool),
            'im': torch.zeros((chip_size, chip_size), dtype=torch.uint8),
        }
    elif task_type == 'bin_segment':
        return {
            'valid': torch.tensor(0, dtype=torch.int32),
            'valid_im': torch.zeros((chip_size, chip_size), dtype=torch.bool),
            'im': torch.zeros((len(task['categories']), chip_size, chip_size), dtype=torch.uint8),
        }
    elif task_type == 'detect' or task_type == 'instance':
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        class_labels = torch.zeros((1,), dtype=torch.int64)
        valid = 0
        target = {}
        target["valid"] = torch.tensor(valid, dtype=torch.int32)
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        if task_type == 'instance':
            target["masks"] = torch.zeros((0, chip_size, chip_size), dtype=torch.float32)
        return target
    elif task_type == 'classification':
        return {
            'label': torch.tensor([0], dtype=torch.int32),
            'valid': torch.tensor(0, dtype=torch.int32),
        }
    elif task_type == 'multi-label-classification':
        return {
            'labels': torch.zeros((1, len(task['categories'])), dtype=torch.int32),
            'valid': torch.tensor(0, dtype=torch.int32),
        }

def load_segment_target(spec, fname):
    task = tasks[spec['Name']]

    if not os.path.exists(fname):
        return get_invalid_target(task)

    im = skimage.io.imread(fname)
    assert len(im.shape) == 2
    valid_im = torch.ones(im.shape, dtype=torch.bool)

    if 'ClassMap' in spec:
        new_im = np.zeros(im.shape, dtype='uint8')
        for cls_name, new_idx in spec['ClassMap'].items():
            old_idx = tasks[spec['Name']]['categories'].index(cls_name)
            new_im[im == old_idx] = new_idx
        im = new_im

    if task.get('BackgroundInvalid', False):
        valid_im &= im > 0

    return {
        'valid': torch.tensor(1, dtype=torch.int32),
        'valid_im': valid_im,
        'im': torch.as_tensor(im),
    }

def load_bin_segment_target(spec, fname):
    task = tasks[spec['Name']]
    is_rgb = task.get('RGB', False)

    # Convert 0-255 mask to binary segmentation labels.
    # The mask is up to 8 bits.
    def mask_to_bin(raw_im, num_categories=None):
        if num_categories is None:
            num_categories = len(task['categories'])
        im = satlas.util.decode_multiclass_binary(raw_im, num_classes=num_categories)
        return im.transpose(2, 0, 1).astype(np.uint8)

    if not os.path.exists(fname):
        return get_invalid_target(task, span)

    label = fname.split('.png')[0]
    if is_rgb:
        im = np.zeros((len(task['categories']), chip_size, chip_size), dtype=np.uint8)
        for i in range((len(task['categories'])+23)//24):
            if i == 0:
                cur_fname = fname
            else:
                cur_fname = '{}_{}.png'.format(label, i)
            rgb_im = skimage.io.imread(cur_fname).astype(np.uint32)
            raw_im = np.zeros((rgb_im.shape[0], rgb_im.shape[1]), dtype=np.uint32)
            raw_im[:, :] |= rgb_im[:, :, 0]
            raw_im[:, :] |= rgb_im[:, :, 1] << 8
            raw_im[:, :] |= rgb_im[:, :, 2] << 16

            num_categories = min(24, len(task['categories']) - i*24)
            im[i*24:(i+1)*24, :, :] = mask_to_bin(raw_im, num_categories=num_categories)
    else:
        raw_im = skimage.io.imread(fname)
        im = mask_to_bin(raw_im)
    valid_im = torch.ones((im.shape[1], im.shape[2]), dtype=torch.bool)

    if 'ClassMap' in spec:
        new_im = np.zeros(im.shape, dtype='uint8')
        for cls_name, new_idx in spec['ClassMap'].items():
            old_idx = tasks[spec['Name']]['categories'].index(cls_name)
            new_im[old_idx, :, :] = 0
            new_im[new_idx, :, :] |= im[old_idx, :, :]
        im = new_im

    if task.get('BackgroundInvalid', False):
        valid_im &= im.max(axis=0) > 0

    return {
        'valid': torch.tensor(1, dtype=torch.int32),
        'valid_im': valid_im,
        'im': torch.as_tensor(im),
    }

def load_regress_target(spec, fname):
    return load_segment_target(spec, fname=fname)

def load_detect_target(spec, fname):
    task = tasks[spec['Name']]
    categories = task['categories']

    boxes = []
    class_labels = []
    valid = 1

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            for x in json.load(f):
                min_x, min_y, max_x, max_y, class_label = x[0:5]
                boxes.append((min_x, min_y, max_x, max_y))
                class_labels.append(categories.index(class_label))
    else:
        valid = 0

    if 'ClassMask' in spec:
        okay_class_ids = set()
        for cls_name in spec['ClassMask']:
            okay_class_ids.add(categories.index(cls_name))
        valid_indexes = [i for i, label in enumerate(class_labels) if label in okay_class_ids]
        boxes = [box for i, box in enumerate(boxes) if i in valid_indexes]
        class_labels = [label for i, label in enumerate(class_labels) if i in valid_indexes]

    if len(boxes) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        class_labels = torch.zeros((1,), dtype=torch.int64)
    else:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_labels = torch.as_tensor(class_labels, dtype=torch.int64)

    target = {}
    target["valid"] = torch.tensor(valid, dtype=torch.int32)
    target["boxes"] = boxes
    target["labels"] = class_labels
    target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
    return target

def load_instance_target(spec, fname):
    task = tasks[spec['Name']]
    categories = task['categories']

    valid = 1
    polygons = []

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            for (polygon_id, coords, class_label, properties) in json.load(f):
                polygons.append((coords, (0, 0), categories.index(class_label)))
    else:
        valid = 0

    instances = []

    for (coords, offset, category_id) in polygons:
        # First element of coords is the polygon exterior ring,
        # rest (if any) are holes.

        offset = np.array(offset, dtype=np.int32)
        exterior = np.array(coords[0], dtype=np.int32) + offset

        box = [
            np.clip(exterior[:, 0].min(), 0, chip_size),
            np.clip(exterior[:, 1].min(), 0, chip_size),
            np.clip(exterior[:, 0].max(), 0, chip_size),
            np.clip(exterior[:, 1].max(), 0, chip_size),
        ]
        min_box_size = spec.get('MinBoxSize', 3)
        if box[2]-box[0] < min_box_size or box[3]-box[1] < min_box_size:
            continue
        if 'MinBoxPerimeter' in spec and 2*((box[3]-box[1])+(box[2]-box[0])) < spec['MinBoxPerimeter']:
            continue

        mask = torch.zeros((chip_size, chip_size), dtype=torch.float32)
        rows, cols = skimage.draw.polygon(exterior[:, 1], exterior[:, 0], shape=(chip_size, chip_size))
        mask[rows, cols] = 1

        for hole in coords[1:]:
            hole = np.array(hole, dtype=np.int32) + offset
            rows, cols = skimage.draw.polygon(hole[:, 1], hole[:, 0], shape=(chip_size, chip_size))
            mask[rows, cols] = 0

        if np.count_nonzero(mask) == 0:
            continue

        instances.append((box, category_id, mask))

    if 'ClassMask' in spec:
        okay_class_ids = set()
        for cls_name in spec['ClassMask']:
            okay_class_ids.add(categories.index(cls_name))
        instances = [item for item in instances if item[1] in okay_class_ids]

    if 'MaxInstances' in spec and len(instances) > spec['MaxInstances']:
        instances = random.sample(instances, spec['MaxInstances'])

    if len(instances) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        class_labels = torch.zeros((1,), dtype=torch.int64)
        masks = torch.zeros((0, chip_size, chip_size), dtype=torch.float32)
    else:
        boxes = torch.as_tensor([instance[0] for instance in instances], dtype=torch.float32)
        class_labels = torch.as_tensor([instance[1] for instance in instances], dtype=torch.int64)
        masks = torch.stack([instance[2] for instance in instances], dim=0)

    target = {}
    target["valid"] = torch.tensor(valid, dtype=torch.int32)
    target["boxes"] = boxes
    target["labels"] = class_labels
    target["masks"] = masks
    target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
    return target

def load_classification_target(spec, fname):
    '''
    Returns the target dict for a classification task.
    '''
    task = tasks[spec['Name']]
    categories = task['categories']
    label = 0
    valid = 0

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            category = f.readline().strip()
            label = categories.index(category)
            valid = 1

    return {
        'label': torch.tensor([label], dtype=torch.int32),
        'valid': torch.tensor(valid, dtype=torch.int32),
    }

def load_multi_label_classification_target(spec, fname):
    '''
    Returns the target dict for a multi-label classification task.
    '''
    task = tasks[spec['Name']]
    n_categories = len(task['categories'])

    target = {
        'labels': torch.zeros((1, n_categories), dtype=torch.int32),
        'valid': torch.tensor(0, dtype=torch.int32),
    }

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            line = f.readline()
            labels = line.split(',')
            for label in labels:
                label = label.replace(',', '').replace(' ', '')
                if not label.isdigit():
                    continue
                target['labels'][0, int(label)] = 1
            target['valid'] = torch.tensor(1, dtype=torch.int32)

    return target

class Option(object):
    # Corresponds to one training example.
    def __init__(self, task_spec=None, task_idx=None, example_id=None, example_dir=None):
        self.task_spec = task_spec
        self.task_idx = task_idx,
        self.example_id = example_id
        self.example_dir = example_dir

    def __repr__(self):
        return '{}_{}'.format(self.task_spec['Name'], self.example_id)

class Dataset(object):
    def __init__(
        self,
        task_specs, # List of task dicts from the config file.
        transforms=None,
        channels=['tci', 'fake', 'fake'],
        selected_task=None, # Only load tiles with outputs for this task.
        max_tiles=None, # Maximum number of tiles to use. We sample the tiles in a deterministic way so that it is consistent across runs.
        num_images = 1, # Max number of images to load for each example.
        task_transforms=None, # Different transforms for particular tasks.
        phase=None, # Specifies Train/Val/Test for the purpose of looking for split filenames.
    ):

        self.task_specs = task_specs
        self.transforms = transforms
        self.channels = channels
        self.tasks = [tasks[spec['Name']] for spec in self.task_specs]
        self.num_images = num_images
        self.task_transforms = task_transforms

        # Create list of options (examples).
        self.options = []

        for task_idx, spec in enumerate(task_specs):
            task_name = spec['Name']

            label_dir = spec['LabelDir']
            task_dir = os.path.join(label_dir, task_name)

            if selected_task and task_name != selected_task:
                continue

            # Get the set of tiles we want to consider for this task, based on the phase-specific split.
            split_name = phase + "Split" # e.g. TrainSplit, ValSplit, TestSplit
            with open(spec[split_name], 'r') as f:
                tile_set = set([(col, row) for col, row in json.load(f)])

            # Add options.
            cur_options = []
            for example_id in os.listdir(task_dir):
                # First two parts of the directory name always specify the zoom 13 tile.
                parts = example_id.split('_')
                tile = (int(parts[0]), int(parts[1]))
                if tile not in tile_set:
                    continue

                cur_options.append(Option(
                    task_spec=spec,
                    task_idx=task_idx,
                    example_id=example_id,
                    example_dir=os.path.join(task_dir, example_id),
                ))

            print('task {}: loaded {} options'.format(task_name, len(cur_options)))
            self.options.extend(cur_options)

        # Reduce number of examples to specified max_tiles if needed.
        # We sort the tiles by their hash to guarantee it is deterministic.
        if max_tiles and len(self.options) > max_tiles:
            print('sample {} -> {} options'.format(len(self.options), max_tiles))
            self.options.sort(key=lambda option: hashlib.md5(repr(option).encode()).hexdigest())
            self.options = self.options[0:max_tiles]

    def __len__(self):
        return len(self.options)

    def __getitem__(self, idx):
        option = self.options[idx]

        # (1) Load images.
        # We sample up to num_images randomly, then sort them by time.
        # If there's an anchor image, we need to always include that at the end of the image time series.
        image_names = []
        anchor_image_name = None
        for image_name in os.listdir(os.path.join(option.example_dir, 'images')):
            if image_name.startswith('anchor_'):
                anchor_image_name = image_name
            else:
                image_names.append(image_name)

        max_sample_images = self.num_images
        if anchor_image_name is not None:
            max_sample_images -= 1
        if len(image_names) > max_sample_images:
            image_names = random.sample(image_names, max_sample_images)
        image_names.sort()
        if anchor_image_name is not None:
            image_names.append(anchor_image_name)
        info = {
            'example_id': option.example_id,
            'image_names': image_names,
            'task_idx': option.task_idx,
            'task_name': option.task_spec['Name'],
        }

        images = []
        for image_name in image_names:
            cur_image = np.zeros((chip_size, chip_size, len(self.channels)), dtype='uint8')

            for channel_idx, channel in enumerate(self.channels):
                if channel == 'fake':
                    continue

                prefix = os.path.join(option.example_dir, 'images', image_name, channel)

                if os.path.exists(prefix+'.png'):
                    fname = prefix+'.png'
                elif os.path.exists(prefix+'.jpg'):
                    fname = prefix+'.jpg'

                im = skimage.io.imread(fname)
                if len(im.shape) == 3 and im.shape[2] == 3:
                    cur_image[:, :, channel_idx:channel_idx+3] = im
                else:
                    cur_image[:, :, channel_idx] = im

            images.append(cur_image.transpose(2, 0, 1))

        while len(images) < self.num_images:
            images = [np.zeros(images[0].shape, dtype=np.uint8)] + images

        data = np.concatenate(images, axis=0)
        data = torch.as_tensor(data)

        # Populate targets.
        targets = []
        for task_idx, spec in enumerate(self.task_specs):
            task_name = spec['Name']
            task = tasks[task_name]
            task_type = task['type']

            if task_name != option.task_spec['Name']:
                targets.append(get_invalid_target(task))
                continue

            if task_type == 'bin_segment':
                fname = os.path.join(option.example_dir, 'gt.png')
                targets.append(load_bin_segment_target(spec, fname=fname))
            elif task_type == 'segment':
                fname = os.path.join(option.example_dir, 'gt.png')
                targets.append(load_segment_target(spec, fname=fname))
            elif task_type == 'regress':
                fname = os.path.join(option.example_dir, 'gt.png')
                targets.append(load_regress_target(spec, fname=fname))
            elif task_type == 'detect':
                fname = os.path.join(option.example_dir, 'gt.json')
                targets.append(load_detect_target(spec, fname=fname))
            elif task_type == 'instance':
                fname = os.path.join(option.example_dir, 'gt.json')
                targets.append(load_instance_target(spec, fname=fname))
            elif task_type == 'classification':
                fname = os.path.join(option.example_dir, 'gt.txt')
                targets.append(load_classification_target(spec, fname=fname))
            elif task_type == 'multi-label-classification':
                fname = os.path.join(option.example_dir, 'gt.txt')
                targets.append(load_multi_label_classification_target(spec, fname=fname))

        # Apply transforms while checking for the self.task_transforms override.
        task_name = option.task_spec['Name']
        if self.task_transforms and task_name in self.task_transforms:
            data, targets = self.task_transforms[task_name](data, targets)
        elif self.transforms:
            data, targets = self.transforms(data, targets)

        return data, targets, info

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for option in self.options:
            option_name = '{}_{}'.format(option.task_spec['Name'], option.example_id)
            weights.append(tile_weights[option_name])

        print('using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        return torch.utils.data.WeightedRandomSampler(weights, len(self.options))
