import numpy as np
import glob, os, sys
from skimage import io, transform

import lmdb
import h5py

sys.path.append('..')
from deepgraph.utils.common import batch

DB_TYPE = 'hdf5'

# TRAIN LMDB prefix
TRAIN_PREFIX = '/Users/sebastian/Desktop/voc_train'


# Generated colormap from PascalVOC2012 dev-kit
COLOR_MAP = np.array([
    [0, 0, 0],
    [0.5020, 0, 0],
    [0, 0.5020, 0],
    [0.5020, 0.5020, 0],
    [0, 0, 0.5020],
    [0.5020, 0, 0.5020],
    [0, 0.5020, 0.5020],
    [0.5020, 0.5020, 0.5020],
    [0.2510, 0, 0],
    [0.7529, 0, 0],
    [0.2510, 0.5020, 0],
    [0.7529, 0.5020, 0],
    [0.2510, 0, 0.5020],
    [0.7529, 0, 0.5020],
    [0.2510, 0.5020, 0.5020],
    [0.7529, 0.5020, 0.5020],
    [0, 0.2510, 0],
    [0.5020, 0.2510, 0],
    [0, 0.7529, 0],
    [0.5020, 0.7529, 0],
    [0.5020,    0.7529,         0],
    [ 0.75294118 , 0.87843137  , 0.87843137] # VOID
])

# Dataset root
DATASET_PATH = '/Users/sebastian/Downloads/VOCdevkit/VOC2012/'

# Load train images
train_img_list = DATASET_PATH + 'ImageSets/Segmentation/trainval.txt'
lines = [line.rstrip('\n') for line in open(train_img_list)]


if DB_TYPE == 'lmdb':
    # Delete DB if exists
    TRAIN_DB_IMAGES = TRAIN_PREFIX + 'images.lmdb'
    TRAIN_DB_LABELS = TRAIN_PREFIX + 'labels.lmdb'
    try:
        os.remove(TRAIN_DB_IMAGES)
    except OSError:
        pass
    try:
        os.remove(TRAIN_DB_LABELS)
    except OSError:
        pass
    # Iterate over those filenames, read image and store it in LMDB
    CHUNK_SIZE = 100
    print "INFO - Iterating files with a chunk size of %i" % CHUNK_SIZE
    imdb = lmdb.open('TRAIN_DB_IMAGES', map_size=1e12)
    ldb = lmdb.open('TRAIN_DB_IMAGES', map_size=1e12)
    for chunk in batch(lines, CHUNK_SIZE):
        # Write images
        with imdb.begin(write=True) as txn:
            idx = 0
            for line in chunk:
                image = np.array(io.imread(DATASET_PATH + 'JPEGImages/' + line + ".jpg"))

        for line in chunk:
            label = np.array(io.imread(DATASET_PATH + 'SegmentationClass/' + line + ".png"))
else:
    DB_FILE = TRAIN_PREFIX + '.hdf5'
    f = h5py.File(DB_FILE)
    # Read images into memory
    """
    print "Starting to process images"
    idx = 0
    images = np.zeros((len(lines), 3, 255, 255))
    for line in lines:
        print "Processing image %i out of %i" % (idx, len(lines))
        image = np.array(io.imread(DATASET_PATH + 'JPEGImages/' + line + ".jpg"))
        rs = transform.resize(image, (255, 255))
        images[idx] = rs.transpose((2,0,1))
        idx += 1
    print "Storing images"
    dimg = f.create_dataset("images", images.shape, dtype=images.dtype)
    dimg[...] = images
    del images
    """
    print "Starting to process labels"
    idx = 0
    images = np.zeros((len(lines), 1, 255, 255))
    for line in lines:
        print "Processing label %i out of %i" % (idx, len(lines))
        image = np.array(io.imread(DATASET_PATH + 'SegmentationClass/' + line + ".png"))
        # Important: Use order=0 to prevent interpolation between adjacent pixels since we need discrete values
        rs = transform.resize(image, (255, 255), order=0)
        # Fill in the label index to produce a 1,255,255 image
        cm = COLOR_MAP.copy()
        image = np.zeros((1,255,255), dtype=np.uint8)
        for y in range(rs.shape[0]):
            for x in range(rs.shape[1]):
                cm = COLOR_MAP.copy()
                px = rs[y, x]
                cm -= px
                indices = np.linalg.norm(cm, axis=1)
                label = np.argmin(indices)
                # Make sure we are hitting a vector in the colormap
                image[0, y, x] = label
        images[idx] = image
        if idx == 5:
            import matplotlib.pyplot as plt
            plt.imshow(image.squeeze())
            plt.show()
        idx += 1
    print "Storing labels"
    dimg = f.create_dataset("labels", images.shape, dtype=images.dtype)
    dimg[...] = images
    del images
    f.close()


