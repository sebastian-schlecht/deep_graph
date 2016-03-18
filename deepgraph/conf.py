import numpy as np

__docformat__ = 'restructedtext en'

rng = np.random.RandomState(1234)   # Globally seed the random generator for reproducible experiments

LOG_LEVEL = 5


def set_log_level(lvl):
    LOG_LEVEL = lvl