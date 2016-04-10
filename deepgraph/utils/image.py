from scipy.misc import imresize
# import cv2
from skimage import exposure
from skimage import transform
from PIL import Image
import numpy as np
import math

def imscale(img, scale):
    """
    Rescale a single image
    :param img: np.array
    :param scale: Float
    :return: Rescaled image
    """
    ii = imresize(img, scale)
    return np.swapaxes(np.swapaxes(ii, 1, 2), 0, 1)


def imbatchresize(array, scale):
    """
    Resize a batch of images (RGB)
    :param array: np.array
    :param scale: Float
    :return: np.array
    """
    assert array.shape[1] == 3
    assert len(array.shape) == 4
    imgs_sized = np.zeros((array.shape[0], array.shape[1], int(array.shape[2], int(array.shape[3]))))
    for i in range(len(array)):
        ii = imresize(array[i], scale)
        imgs_sized[i] = np.swapaxes(np.swapaxes(ii, 1, 2), 0, 1)


def noise_transformer(image, intensity= 0.1, mean=0, std=0.01):
    """
    Add random noise to a given image
    :param image: np.array
    :param std: Standard deviation
    :return: np.array
    """
    image = image.copy()
    noise = np.random.normal(mean, std, size=image.shape)
    image = image + (noise * intensity)
    return image.astype(np.uint8)



def flip_transformer_s(image, direction):
    """
    Flip a greyscale image/array
    :param image: Image/Array
    :param direction: Flip dir
    :return: NDArray
    """
    if direction is 'h':
        return image[:, ::-1]
    else:
        return image[::-1, :]


def flip_transformer_rgb(image, direction):
    """
    Flit a RGB image alongside an axis
    :param image: np.array
    :param direction: String
    :return: NDArray
    """
    if direction is 'h':
        return image[:, :, ::-1]
    else:
        return image[:, ::-1, :]


def rotate_transformer_scalar_float32(array, angle, normalize=100.):
    """
    Rotate a scalar float image by angle degrees. Normalize is used to force the values in -1,1 range for rotate to work
    Fill empty areas with a 0
    :param array: Arraylike
    :param angle: Float
    :param normalize: Float
    :return: Arraylike
    """
    array = array.copy()
    array /= normalize
    array = transform.rotate(array, angle)
    array *= normalize
    return array


def rotate_transformer_rgb_uint8(array, angle):
    """
    Rotate an RGB image by angle degrees
    Fill empty areas with 0
    :param array: Arraylike
    :param angle: Float
    :return: Arraylike
    """
    array = array.copy()
    array = transform.rotate(array.transpose((1, 2, 0)), angle)
    array *= 255.
    return array.astype(np.uint8).transpose((2, 0, 1))


def exposure_transformer(img, level):
    """
    Change the exposure in an RGB image
    :param img: np.array
    :param level: Number
    :return: Image
    """
    img = Image.fromarray(img.transpose(1,2,0), mode="RGB")

    def truncate(v):
        return 0 if v < 0 else 255 if v > 255 else v
    if Image.isStringType(img):  # file path?
        img = Image.open(img)
    if img.mode not in ['RGB', 'RGBA']:
        raise TypeError('Unsupported source image mode: {}'.format(img.mode))
    img.load()

    factor = (259 * (level+255)) / (255 * (259-level))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            color = img.getpixel((x, y))
            new_color = tuple(truncate(factor * (c-128) + 128) for c in color)
            img.putpixel((x, y), new_color)
    return np.array(img).transpose((2,0,1))


def batch_pad_mirror(tensor, border):
    assert tensor.shape[2] > border
    assert tensor.shape[3] > border
    # Copy core
    replica = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2] + 2*border, tensor.shape[3] + 2*border))
    replica[:, :, border:border + tensor.shape[2], border:border + tensor.shape[3]] = tensor.copy()
    # Mirror top
    replica[:,:,0:border,border:border+tensor.shape[3]] = tensor[:,:,0:border,:][:,:,::-1,:]
    # Bottom
    replica[:, :, tensor.shape[2]+border:, border:border+tensor.shape[3]] = tensor[:, :, tensor.shape[2]-border:, :][:, :, ::-1, :]
    # Left. Use itself for mirroring
    replica[:, :, :, 0:border] = replica[:, :, :, border:2*border][:, :, :, ::-1]
    # Right
    replica[:, :, :, tensor.shape[3]+border:tensor.shape[3]+2*border] = \
        replica[:, :, :, tensor.shape[3]:tensor.shape[3]+border][:, :, :, ::-1]
    return replica
