import threading
import Queue
import time
import math

import theano
import numpy as np
import h5py
from scipy.misc import imresize

from deepgraph.utils.logging import log
from deepgraph.utils.common import batch_parallel
from deepgraph.constants import *


class Pipeline(object):
    """
    Pipeline class to assemble processor units
    """
    SIG_FINISHED = 0
    SIG_ABORT = -1

    def __init__(self, config):
        """
        Constructor
        :return: Pipeline
        """
        self.processors = []
        self.last_item = None
        self.config = config

    def add(self, processor):
        """
        Add a new processor on top of the stack. Processors will be connected in call order e.g.

        pipeline.add(db)
        pipeline.add(optimizer)

        would set optimizer as the top stage of db, thus receiving data chunks from the db loader
        :param processor: Processor
        :return: None
        """
        self.processors.append(processor)
        if self.last_item is not None:
            self.last_item.top = processor
            processor.bottom = self.last_item
        self.last_item = processor
        processor.pipeline = self

    def run(self):
        """
        Start running processors
        :return: None
        """
        for proc in self.processors:
            proc.start()

    def stop(self):
        """
        Stop running processors
        :return: None
        """
        for proc in self.processors:
            proc.stop.set()

    def signal(self, message):
        """
        Receive a signal from child processors
        :param message:
        :return:
        """
        if message == Pipeline.SIG_FINISHED:
            self.stop()
        if message == Pipeline.SIG_ABORT:
            log("Abort signal sent. Pipeline stopping.", LOG_LEVEL_WARNING)





class Packet(object):
    """
    Packet container for generic data flowing through the pipeline
    """
    def __init__(self, id, phase, shapes, data):
        """
        Constructor
        :param phase: Int (Phase)
        :param shapes: Tuple
        :param data: AnyType
        :return:
        """
        self.id = id
        self.phase = phase
        self.shapes = shapes
        self.data = data


class Processor(threading.Thread):
    """
    Generic processor class which implements a stage in the pipeline. Each processor may have at max
    one predecessor and one successor. A processor has to wait until the top level stage is been initialized.
    That guarantees that data is passing forward only after all stages have been through their init phase.
    Each processor has an entry queue where other parts of the pipeline can put data in
    """
    SPIN_WAIT_TIME = 0.5    # Wait time in case process block has not done any work
    QUEUE_FULL_OR_EMPTY_TIMEOUT = 1     # Timeout after which stop states are checked

    def __init__(self, name, shapes, config, buffer_size=10):
        """
        Constructor
        :param name: String
        :param shapes: Tuple of tuples
        :param config: Dict
        :param buffer_size: Int
        :return:
        """
        super(Processor, self).__init__(group=None)
        self.top = None
        self.bottom = None
        self.name = name
        self.shapes = shapes
        self.config = config
        self.is_consumer = True
        self.is_producer = True
        self.q = Queue.Queue(buffer_size)
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.stop.clear()
        self.pipeline = None

    def run(self):
        """
        Override thread run method. Run is being called automatically when the thread has been started via start()
        :return:
        """
        self.init()
        if self.top is not None:
            self.top.ready.wait()
        self.ready.set()
        while not self.stop.is_set():
            res = self.process()
            if not res:
                time.sleep(Processor.SPIN_WAIT_TIME)

    def init(self):
        """
        Init routine. Can be left empty. In this section necessary data setups can take place (graph compilation,
        opening database handles, allocating memory
        :return:
        """
        pass

    def process(self):
        """
        Mandatory method which is called in the run loop. If this method returns True payload has been processed.
        In case False is returned, the thread will wait for SPIN_WAIT_TIME seconds to re-enter the processing block
        :return: Bool
        """
        raise NotImplementedError("Abstract method process has to be implemented")


class H5DBLoader(Processor):
    """
    Processor class to asynchronously load data in chunks and prepare it for later stages
    TODO Move example specific code out of this class
    """
    def __init__(self, name, shapes, config, buffer_size=10):
        super(H5DBLoader, self).__init__(name, shapes, config, buffer_size)
        self.db_handle = None
        self.cursor = 0
        self.data_field = None
        self.label_field = None

    def init(self):
        """
        Open a handle to the database and check if the config is valid and files exist
        :return:
        """
        assert "db" in self.config
        self.db_handle = h5py.File(self.config["db"])
        assert "key_data" in self.config
        assert "key_label" in self.config
        assert "chunk_size" in self.config
        self.data_field = self.db_handle[self.config["key_data"]]
        self.label_field = self.db_handle[self.config["key_label"]]

    def process(self):
        """
        Read elements from the database in packes specified by "chunk_size". Feed each packet to the top processor
        in case its entry queue is not full
        :return: Bool
        """
        if self.top is not None:
            log("H5DBLoader: Preloading chunk", LOG_LEVEL_VERBOSE)
            c_size = self.config["chunk_size"]
            start = time.time()
            data = self.data_field[self.cursor:self.cursor+c_size]
            label = self.label_field[self.cursor:self.cursor+c_size]

            # Try to push into queue as long as thread should not terminate
            while not self.stop.is_set():
                try:
                    self.top.q.put((data, label), block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                    break
                except Queue.Full:
                    log("H5DBLoader: Waiting for upstream processor to finish", LOG_LEVEL_VERBOSE)
                    # In case the queue is empty, return false, wait for spin and check if we have to abort
                    pass
            end = time.time()
            log("H5DBLoader: Fetching took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
            return True
        else:
            # No top node found, enter spin wait time
            return False


class Optimizer(Processor):
    """
    Processor implementation of a solver which processes data in chunks and transfers everything to the
    computing device for further iteration. E.g. with batches of size 32, a chunk would be of size 320 (example).
    After transferring the chunk to the GPU (if available) the optimizer loops through that chunk 10 times. After that,
    the next chunk is taken from the entry queue
    """
    def __init__(self, name, graph, shapes, config, buffer_size=10):
        super(Optimizer, self).__init__(name, shapes, config, buffer_size)
        assert graph is not None
        self.graph = graph
        # Shared vars
        self.var_x = None
        self.var_y = None
        self.idx = 0

    def init(self):
        """
        Allocate some shared var's we use to store chunks in. After that, compile the graph so that gradients
        can be computed. We only do training here, no inference so compile in training mode only.
        :return: None
        """
        # Theano shared variables
        self.var_x = theano.shared(np.ones(self.shapes[0], dtype=theano.config.floatX), borrow=False)
        self.var_y = theano.shared(np.ones(self.shapes[1], dtype=theano.config.floatX), borrow=False)
        # Compile
        self.graph.compile(train_inputs=[self.var_x, self.var_y], batch_size=32, phase=PHASE_TRAIN)
        self.idx = 0

    def process(self):
        """
        Take an element out of the entry queue and iterate through it. Mark that an additional call to batch_parallel
        is made in case the chunk_size of this node differs from the producing node.
        :return: Bool (True)
        """
        # Get an element. This blocks if empty which is what we want
        has_data = False
        while not self.stop.is_set():
            try:
                train_x, train_y = self.q.get(block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                has_data = True
                break
            except Queue.Empty:
                log("Optimizer: Waiting for data", LOG_LEVEL_VERBOSE)
                pass
        # Return if no data is there
        if not has_data:
            return False
        start = time.time()
        assert (train_x.shape == self.shapes[0]) and (train_y.shape == self.shapes[1])
        if self.idx < self.config["iters"]:
            for chunk_x, chunk_y in batch_parallel(train_x, train_y, self.config["chunk_size"]):
                log("Optimizer: Transferring data to computing device", LOG_LEVEL_VERBOSE)
                # Assign the super-batch to the shared variable
                self.var_x.set_value(chunk_x, borrow=False)
                self.var_y.set_value(chunk_y, borrow=False)
                # Iterate through the super batch
                n_iters = int(math.ceil(len(chunk_x) / float(self.config["batch_size"])))
                for minibatch_index in range(n_iters):
                    log("Optimizer: Computing gradients", LOG_LEVEL_VERBOSE)
                    self.idx += 1
                    minibatch_avg_cost = self.graph.models[TRAIN](
                        minibatch_index,
                        self.config["learning_rate"],
                        self.config["momentum"],
                        self.config["weight_decay"]
                    )
                    # Print in case the freq is ok
                    if self.idx % 200 == 0:
                        log("Training score at iteration %i: %s" % (self.idx, str(minibatch_avg_cost)), LOG_LEVEL_INFO)
                    # Abort if we reached max iters
                    if self.idx >= self.config["iters"]:
                        break
        # We're done
        else:
            self.pipeline.signal(Pipeline.SIG_FINISHED)
        end = time.time()
        log("Optimizer: Computation took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
        # Return true, we don't want to enter spin waits. Just proceed with the next chunk or stop
        return True


class Transformer(Processor):
    """
    Example augmentation implementation
    """
    def __init__(self, name, shapes, config, buffer_size=10):
        super(Transformer, self).__init__(name, shapes, config, buffer_size)

    def init(self):
        pass

    def process(self):
        # Check if fully connected
        if self.top is None:
            return False
        # Query data
        has_data = False
        while not self.stop.is_set():
            try:
                data, label = self.q.get(block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                has_data = True
                break
            except Queue.Empty:
                log("Transformer: Waiting for data", LOG_LEVEL_VERBOSE)
                pass
        # Return if no data is there
        if not has_data:
            return False

        # Do processing
        log("Transformer: Processing data", LOG_LEVEL_VERBOSE)
        start = time.time()
        data = np.swapaxes(data, 2, 3)
        label = np.swapaxes(label, 1, 2)

        data_scale = 0.5
        label_scale = 0.125

        data_sized = np.zeros((data.shape[0], data.shape[1], int(data.shape[2]*data_scale), int(data.shape[3]*data_scale)), dtype=np.uint8)
        label_sized = np.zeros((label.shape[0], int(label.shape[1]*label_scale), int(label.shape[2]*label_scale)), dtype=np.float32)

        for i in range(len(data)):
            ii = imresize(data[i], data_scale)
            data_sized[i] = np.swapaxes(np.swapaxes(ii, 1, 2), 0, 1)

        # For this test, we down-sample the depth images to 64x48
        for d in range(len(label)):
            dd = imresize(label[d], label_scale)
            label_sized[d] = dd

        data = data_sized
        label = label_sized
        end = time.time()
        log("Transformer: Processing took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
        # Try to push into queue as long as thread should not terminate
        while not self.stop.is_set():
            try:
                self.top.q.put((data, label), block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                break
            except Queue.Full:
                log("Transformer: Waiting for upstream processor to finish", LOG_LEVEL_VERBOSE)
                # In case the queue is empty, return false, wait for spin and check if we have to abort
                pass
        return True



