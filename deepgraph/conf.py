import numpy as np
from deepgraph.utils.logging import *
from deepgraph.constants import *

__docformat__ = 'restructedtext en'

rng = np.random.RandomState(1234)   # Globally seed the random generator for reproducible experiments

LOG_LEVEL = LOG_LEVEL_DEV


def set_log_level(lvl):
    LOG_LEVEL = lvl