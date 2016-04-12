import numpy as np
import theano
from deepgraph.conf import rng

__docformat__ = 'restructedtext en'


def normal(mu=0, dev=0.01, dtype=theano.config.floatX):
    """
    Return a generating function filling weights with a normal distribution
    :param mu: Float
    :param dev: Float
    :param dtype: np.type
    :return:
    """
    def gen(size):
        return np.asarray(rng.normal(mu, dev, size=size), dtype=dtype)
    return gen


def uniform(low=-0.5, high=0.5, dtype=theano.config.floatX):
    """
    Return a generating function filling weights with
    :param low: Float
    :param high: Float
    :param dtype: np.type
    :return:
    """
    def gen(size):
        return np.asarray(rng.uniform(low=low, high=high, size=size), dtype=dtype)
    return gen


def zeros(dtype=theano.config.floatX):
    """
    Zero filled tensor
    :param dtype: np.type
    :return:
    """
    def gen(size):
        return np.zeros(size, dtype=dtype)
    return gen


def constant(value=1, dtype=theano.config.floatX):
    """
    Fill with a constant value (e.g. 1)
    :param value:
    :param dtype: np.type
    :return:
    """
    def gen(size):
        arr = np.empty(size, dtype=dtype)
        arr.fill(value)
        return arr
    return gen

def xavier(gain=1.0, dtype=theano.config.floatX):
    """
    Xavier init according to http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    if gain == "relu":
        gain = np.sqrt(2)
    def gen(size):
        if len(size) < 2:
            raise AssertionError("This initializer only works with shapes of length >= 2")

        n1, n2 = size[:2]
        receptive_field_size = np.prod(size[2:])

        std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return np.asarray(rng.normal(0, std, size=size), dtype=dtype)
        
    return gen