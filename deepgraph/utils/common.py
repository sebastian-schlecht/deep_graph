import numpy as np
import theano.tensor as T
import six.moves.cPickle as pickle


from deepgraph.utils.logging import *
from deepgraph.constants import *


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


def shuffle_in_unison_inplace(a, b):
    """
    Shuffle two arrays parallel
    :param a: Array
    :param b: Array
    :return: Tuple
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


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


def pickle_dump(obj, filename):
    """
    Write an object to a location in pickling format
    :param obj: Any object
    :param filename: String
    :return:
    """
    pickle.dump(obj, open(filename, "wb"))


def pickle_load(filename):
    """
    Load an object in pickling format
    :param filename: String
    :return:
    """
    return pickle.load(open(filename, "rb"))


class ConfigMixin(object):
    """
    Base object to stuff config functionality into the object
    """
    def make_configurable(self, config=None):
        self.__default_values___ = []
        self.__config__ = {}
        self.setup_defaults()
        if config is not None:
            assert isinstance(config, dict)
            # Check keys
            for key in config:
                if key not in self.__default_values___:
                    log("Key '%s' has no default value. Is the spelling correct?" % key, LOG_LEVEL_WARNING)
                self.__config__[key] = config[key]

    def conf(self, key, value=None):
        """
        Get the configuration value
        :param key: String
        :return:
        """
        if value == None:
            if key in self.__config__:
                return self.__config__[key]
            else:
                log("Accessing non standard configuration property", LOG_LEVEL_WARNING)
                return None
        else:
            if key not in self.__config__:
                log("Accessing non standard configuration property", LOG_LEVEL_WARNING)
            self.__config__[key] = value

    def conf_default(self, key, value):
        """
        Set default value
        :param key: String
        :param value: Anytype
        :return:
        """
        self.__default_values___.append(key)
        self.__config__[key] = value

    def setup_defaults(self):
        """
        Abstract method to setup default values. Has to be implemented by child classes
        :return:
        """
        raise NotImplementedError("Abstract method setup_defaults has to be implemented.")

