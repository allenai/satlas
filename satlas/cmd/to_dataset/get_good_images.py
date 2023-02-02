import argparse
import json
import multiprocessing
import numpy
import os
import skimage.io
import tqdm

# Create JSON file listing available images/tiles.
# We exclude "bad" images here, like lots of black (missing) or lots of white (cloud/snow) pixels.

splits = ['test', 'test_highres', 'train', 'train_gold', 'val']
image_types = ['images', 'highres']

def is_good_image(t):
    col, row, image_uuid = t
    im_path = os.path.join(image_dir, image_uuid, 'tci', '{}_{}.png'.format(col, row))
    try:
        im = skimage.io.imread(im_path)
    except Exception as e:
        print('warning: got exception {} loading {}'.format(e, im_path))
        return False
    black_fraction = numpy.count_nonzero(im.max(axis=2) == 0)/im.shape[0]/im.shape[1]
    white_fraction = numpy.count_nonzero(im.min(axis=2) > 230)/im.shape[0]/im.shape[1]
    return black_fraction < 0.2 and white_fraction < 0.2

def get_chips(image_dir):
    chips = []

    for image_uuid in os.listdir(image_dir):
        tci_path = os.path.join(image_dir, image_uuid, 'tci')
        if not os.path.exists(tci_path):
            continue
        for fname in os.listdir(tci_path):
            if not fname.endswith('.png'):
                continue
            im_path = os.path.join(tci_path, fname)
            parts = fname.split('.png')[0].split('_')
            col = int(parts[0])
            row = int(parts[1])
            chips.append((col, row, image_uuid))

    return chips

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get good images.")
    parser.add_argument("--satlas_root", help="Satlas root directory.")
    parser.add_argument("--workers", help="Number of worker processes.", type=int, default=32)
    args = parser.parse_args()

    for split in splits:
        for image_type in image_types:
            image_dir = os.path.join(args.satlas_root, split, image_type)
            out_fname = os.path.join(args.satlas_root, 'metadata', 'good_{}_{}.json'.format(image_type, split))

            print('processing', image_dir)
            all_chips = get_chips(image_dir)

            p = multiprocessing.Pool(args.workers)
            good = list(tqdm.tqdm(p.imap(is_good_image, all_chips), total=len(all_chips)))
            p.close()
            good_chips = [chip for i, chip in enumerate(all_chips) if good[i]]

            with open(out_fname, 'w') as f:
                json.dump(good_chips, f)
