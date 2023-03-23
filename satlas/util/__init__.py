import json
import math
import numpy as np
import os
import skimage.io

def clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def geo_to_mercator(p, zoom=13, pixels=512):
	n = 2**zoom
	x = (p[0] + 180.0) / 360 * n
	y = (1 - math.log(math.tan(p[1] * math.pi / 180) + (1 / math.cos(p[1] * math.pi / 180))) / math.pi) / 2 * n
	return (x*pixels, y*pixels)

def mercator_to_geo(p, zoom=13, pixels=512):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)

def load_split(fname, image_list_ok=False):
    # Load a split.
    # If image_list_ok is set, the input may either be a split or an image list.
    with open(fname, 'r') as f:
        split = json.load(f)
    if len(split[0]) == 2:
        return [(col, row) for col, row in split]
    elif len(split[0]) == 3 and image_list_ok:
        return [(col, row) for col, row, _ in split]
    else:
        raise Exception('bad format in {}'.format(fname))

def interpolate_corners(tile, pixels=512, zoom=13):
    c1 = mercator_to_geo(tile, zoom=17, pixels=1)  # upper left
    c2 = mercator_to_geo((tile[0] * pixels + pixels, tile[1]), zoom=17, pixels=pixels)  # upper right
    c3 = mercator_to_geo((tile[0], tile[1] * pixels - pixels), zoom=17, pixels=pixels)  # bottom left
    c4 = mercator_to_geo((tile[0] * pixels + pixels, tile[1] * pixels - pixels), zoom=17, pixels=pixels)  # bottom right
    a=np.linspace(c1, c2, 512)
    b=np.linspace(c3, c4, 512)
    c = np.array([np.linspace(i, j, 512) for i,j in zip(a,b)])  # [512, 512, 2]
    return c[:, :, 0], c[:, :, 1]

# Convert binary [H, W, C] to/from a uint32 [H, W].
def encode_multiclass_binary(arr):
    enc = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint64)
    for cls_id in range(arr.shape[2]):
        enc[arr[:, :, cls_id]] |= np.uint64(2**cls_id)
    return enc

def decode_multiclass_binary(enc, num_classes):
    arr = np.zeros((enc.shape[0], enc.shape[1], num_classes), dtype=bool)
    for cls_id in range(num_classes):
        arr[:, :, cls_id] = (enc & (2**cls_id)) > 0
    return arr

def load_window(base_dir, column, row, width, height, chip_size=512, bands=3):
    im = np.zeros((height, width, bands), dtype=np.uint8)

    # Load tiles one at a time.
    start_tile = (column//chip_size, row//chip_size)
    end_tile = ((column+width-1)//chip_size, (row+height-1)//chip_size)
    for i in range(start_tile[0], end_tile[0]+1):
        for j in range(start_tile[1], end_tile[1]+1):
            fname = os.path.join(base_dir, '{}_{}.png'.format(i, j))
            if not os.path.exists(fname):
                continue

            cur_im = skimage.io.imread(fname)
            if bands == 1 and len(cur_im.shape) == 2:
                # Add channel dimension for greyscale images.
                cur_im = cur_im[:, :, None]

            cur_col_off = chip_size*i
            cur_row_off = chip_size*j

            src_col_offset = max(column - cur_col_off, 0)
            src_row_offset = max(row - cur_row_off, 0)
            dst_col_offset = max(cur_col_off - column, 0)
            dst_row_offset = max(cur_row_off - row, 0)
            col_overlap = min(cur_im.shape[1] - src_col_offset, width - dst_col_offset)
            row_overlap = min(cur_im.shape[0] - src_row_offset, height - dst_row_offset)
            im[dst_row_offset:dst_row_offset+row_overlap, dst_col_offset:dst_col_offset+col_overlap] = cur_im[src_row_offset:src_row_offset+row_overlap, src_col_offset:src_col_offset+col_overlap]

    return im