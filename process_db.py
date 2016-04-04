import numpy as np, h5py
from scipy.misc import imresize
from scipy.ndimage import zoom

db_file = "./data/nyu_depth_v2_labeled.mat"
db_file_2 = "./data/nyu_depth_v1_labeled.mat"
target_file = "./data/nyu_depth_combined.hdf5"


# Read the MAT-File into memory
dataset = h5py.File(db_file)

depth_field = dataset['depths']
depths = np.array(depth_field)

images_field = dataset['images']
images = np.array(images_field).astype(np.uint8)

# Swap axes
images = np.swapaxes(images, 2, 3)
depths = np.swapaxes(depths, 1, 2)

# Resizing
img_scale = 0.6
depth_scale = 0.15

images_sized = np.zeros((images.shape[0], images.shape[1], int(images.shape[2]*img_scale), int(images.shape[3]*img_scale)), dtype=np.uint8)
depths_sized = np.zeros((depths.shape[0], int(depths.shape[1]*depth_scale), int(depths.shape[2]*depth_scale)), dtype=np.float32)

for i in range(len(images)):
    ii = imresize(images[i], img_scale)
    images_sized[i] = np.swapaxes(np.swapaxes(ii, 1, 2), 0, 1)

# For this test, we down-sample the depth images to 64x48

for d in range(len(depths)):
    dd = zoom(depths[d], depth_scale)
    depths_sized[d] = dd

images_1 = images_sized
depths_1 = depths_sized


# Now we do the same thing for the second db file
dataset = h5py.File(db_file_2)

depth_field = dataset['depths']
depths = np.array(depth_field)

images_field = dataset['images']
images = np.array(images_field).astype(np.uint8)

# Swap axes
images = np.swapaxes(images, 2, 3)
depths = np.swapaxes(depths, 1, 2)

# Resizing
img_scale = 0.6
depth_scale = 0.15

images_sized = np.zeros((images.shape[0], images.shape[1], int(images.shape[2]*img_scale), int(images.shape[3]*img_scale)), dtype=np.uint8)
depths_sized = np.zeros((depths.shape[0], int(depths.shape[1]*depth_scale), int(depths.shape[2]*depth_scale)), dtype=np.float32)

for i in range(len(images)):
    ii = imresize(images[i], img_scale)
    images_sized[i] = np.swapaxes(np.swapaxes(ii, 1, 2), 0, 1)

# For this test, we down-sample the depth images to 64x48

for d in range(len(depths)):
    dd = zoom(depths[d], depth_scale)
    depths_sized[d] = dd

images = images_sized
depths = depths_sized

images = np.concatenate([images, images_1])
depths = np.concatenate([depths, depths_1])

# Persist
f = h5py.File(target_file, "w")
depths_f = f.create_dataset("depths", depths.shape, dtype=depths.dtype)
depths_f[...] = depths

images_f = f.create_dataset("images", images.shape, dtype=images.dtype)
images_f[...] = images

f.close()
