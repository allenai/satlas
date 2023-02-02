import datetime
import json
import os
import tqdm

image_types = [{
    'name': 'images',
    'label': 'sentinel2',
    'factor': 1,
}, {
    'name': 'highres',
    'label': 'naip',
    'factor': 16,
}]

def get_image_types(satlas_root, split):
    for image_meta in image_types:
        print('preprocess image type', image_meta['name'])
        image_times_fname = os.path.join(satlas_root, 'metadata', 'image_times_{}.json'.format(image_meta['label']))
        image_times = {}
        with open(image_times_fname, 'r') as f:
            for image_uuid, ts_str in json.load(f).items():
                ts = datetime.datetime.fromisoformat(ts_str)
                image_times[image_uuid] = ts
        image_meta['image_times'] = image_times

        image_list_fname = os.path.join(satlas_root, 'metadata', 'good_{}_{}.json'.format(image_meta['name'], split))
        images_by_tile = {}
        with open(image_list_fname, 'r') as f:
            for col, row, image_uuid in json.load(f):
                if image_uuid not in image_times:
                    continue
                if (col, row) not in images_by_tile:
                    images_by_tile[(col, row)] = []
                images_by_tile[(col, row)].append(image_uuid)
        image_meta['image_list'] = images_by_tile

    return image_types

# Get image UUID that covers a tile and falls within specified time range.
def get_image_uuids(image_meta, tile, ts1, ts2):
    image_times = image_meta['image_times']
    image_uuids = []
    for image_uuid in image_meta['image_list'].get(tile, []):
        image_ts = image_times[image_uuid]
        if image_ts < ts1 or image_ts > ts2:
            continue
        image_uuids.append(image_uuid)
    return image_uuids
