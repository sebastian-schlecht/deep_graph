import numpy as np
import theano
import theano.tensor as T


def batch(iterable, n=1):
    """
    Split an array into batches (for example for writing them to a DB batch-wise

    Example:    'for x in batch(array, 3):' for iterating 'array' in batches of 3 elements.

    :param iterable: The array to be split
    :param n: Batch-size
    :return: Iterable
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def batch_parallel(iterable_a, iterable_b, n=1):
    """
    Split an array into batches (for example for writing them to a DB batch-wise
    Example:    'for x in batch(array, 3):' for iterating 'array' in batches of 3 elements.
    :param iterable_a: The first array to be split
    :param iterable_b: The second array to be split
    :param n: Batch-size
    :return: Iterable
    """
    assert(len(iterable_a) == len(iterable_b))
    l = len(iterable_a)
    for ndx in range(0, l, n):
        yield iterable_a[ndx:min(ndx + n, l)], iterable_b[ndx:min(ndx + n, l)]


def split_array(array, ratio=0.9):
    """
    Split an array according to a given ratio along the first axis. Useful to partition arrays into train/val
    :param array: The array to split
    :param ratio: The ratio to split with
    :return:
    """
    assert ratio > 0
    assert ratio < 1
    return (array[0:int(ratio*len(array))], array[int(ratio*len(array)):])


def wrap_shared(array, borrow=True, cast=None):
    """
    Wrap an array into a theano shared var for complete transfer to the GPU
    :param array: List or ndarray
    :param borrow: Boolean
    :param cast: String
    :return:
    """
    data = theano.shared(np.asarray(array, dtype=theano.config.floatX), borrow=borrow)
    if cast is None:
        return data
    else:
        return T.cast(data, cast)

