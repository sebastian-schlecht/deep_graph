import time
import datetime

from deepgraph.conf import LOG_LEVEL

LOG_LEVEL_ERROR = 0
LOG_LEVEL_WARNING = 1
LOG_LEVEL_INFO = 2
LOG_LEVEL_DEV = 3
LOG_LEVEL_VERBOSE = 4

LOG_LEVEL_STRINGS = ["ERROR", "WARNING", "INFO", "DEV", "VERBOSE"]


def log(string, log_level):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    if LOG_LEVEL >= log_level:
        print("[" + st + "] " + LOG_LEVEL_STRINGS[log_level] + ": " + string)
