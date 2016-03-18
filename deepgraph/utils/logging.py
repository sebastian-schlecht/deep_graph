import time
import datetime

from deepgraph.conf import LOG_LEVEL


def log(string, log_level):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    if LOG_LEVEL > log_level:
        print("[" + st + "] " + string)