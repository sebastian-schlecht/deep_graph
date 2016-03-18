import numpy as np
from deepgraph.utils.logging import *

__docformat__ = 'restructedtext en'

rng = np.random.RandomState(1234)   # Globally seed the random generator for reproducible experiments

LOG_LEVEL = 4

def set_log_level(lvl):
    LOG_LEVEL = lvl