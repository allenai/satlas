import datetime
import json
import os
import tqdm

image_types = [{
    'name': 'lowres',
    'label': 'sentinel2',
    'factor': 1,
}, {
    'name': 'highres',
    'label': 'naip',
    'factor': 16,
}]

def get_image_types(satlas_root):
    for image_meta in image_types:
        print('preprocess image type', image_meta['name'])
        image_times_fname = os.path.join(satlas_root, 'metadata', 'image_times.json')
        image_times = {}
        with open(image_times_fname, 'r') as f:
            for image_uuid, ts_str in json.load(f).items():
                ts = datetime.datetime.fromisoformat(ts_str)
                image_times[image_uuid] = ts
        image_meta['image_times'] = image_times

        image_list_fname = os.path.join(satlas_root, 'metadata', 'good_images_{}_all.json'.format(image_meta['name']))
        images_by_tile = {}
        with open(image_list_fname, 'r') as f:
            for col, row, image_uuid in json.load(f):
                if image_uuid not in image_times:
                    continue
                if not os.path.exists(os.path.join(satlas_root, image_meta['label'], image_uuid, 'tci', '{}_{}.png'.format(col, row))):
                    continue
                if (col, row) not in images_by_tile:
                    images_by_tile[(col, row)] = []
                images_by_tile[(col, row)].append(image_uuid)
        image_meta['image_list'] = images_by_tile

    return image_types