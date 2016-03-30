import threading
import Queue
import time
import math

import theano
import numpy as np
import h5py
from scipy.misc import imresize
from scipy.ndimage import zoom


from deepgraph.utils.logging import log
from deepgraph.utils.common import batch_parallel
from deepgraph.constants import *
from deepgraph.conf import rng


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
        # TODO Feed pipeline with instructions

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
            self.stop()


class Packet(object):
    """
    Packet container for generic data flowing through the pipeline
    """
    def __init__(self, identifier, phase, num, shapes, data):
        """
        Constructor
        :param identifier: Packet identifier
        :param phase: Phase identifier
        :param num: Number of tensors inside the packet
        :param shapes: Shapes for each tensor
        :param data: Data to transport
        :return:
        """
        self.id = identifier
        self.phase = phase
        self.num = num
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
        try:
            self.init()
            if self.top is not None:
                self.top.ready.wait()
            self.ready.set()
            while not self.stop.is_set():
                res = self.process()
                if not res:
                    time.sleep(Processor.SPIN_WAIT_TIME)
        # In case this loop catches any exception we abort the whole process
        except Exception as e:
            self.pipeline.signal(Pipeline.SIG_ABORT)
            raise e


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

    def pull(self):
        """
        Pull data from previous processor
        :return:
        """
        has_data = False
        rval = None
        while not self.stop.is_set():
            try:
                rval = self.q.get(block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                has_data = True
                break
            except Queue.Empty:
                pass
        # Return if no data is there
        if not has_data:
            return None
        return rval

    def push(self, data):
        """
        Push data into next processor
        :param data:
        :return:
        """
        while not self.stop.is_set():
            try:
                self.top.q.put(data, block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                break
            except Queue.Full:
                # In case the queue is empty, return false, wait for spin and check if we have to abort
                pass


class H5DBLoader(Processor):
    """
    Processor class to asynchronously load data in chunks and prepare it for later stages
    """
    def __init__(self, name, shapes, config, buffer_size=10):
        super(H5DBLoader, self).__init__(name, shapes, config, buffer_size)
        self.db_handle = None
        self.cursor = 0
        self.data_field = None
        self.label_field = None
        self.thresh = 0

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
        assert self.data_field.shape[0] == self.label_field.shape[0]
        split = self.config["split_value"] if "split_value" in self.config else 0.9
        self.thresh = int(split * self.data_field.shape[0])

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
            upper = min(self.thresh, self.cursor + c_size)
            data = self.data_field[self.cursor:upper]
            label = self.label_field[self.cursor:upper]

            self.cursor += c_size
            # Reset cursor in case we exceeded the array ranges
            if self.cursor > self.thresh:
                self.cursor = 0

            self.push((data, label))
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
        # Load weights if necessary
        if "weights" in self.config:
            self.graph.load_weights(self.config["weights"])
        # Compile
        self.graph.compile(train_inputs=[self.var_x, self.var_y], batch_size=self.config["batch_size"], phase=PHASE_TRAIN)
        self.idx = 0

    def process(self):
        """
        Take an element out of the entry queue and iterate through it. Mark that an additional call to batch_parallel
        is made in case the chunk_size of this node differs from the producing node.
        :return: Bool (True)
        """
        data = self.pull()
        # Return if no data is there
        if not data:
            return False
        train_x, train_y = data
        start = time.time()
        assert (train_x.shape[1:] == self.shapes[0][1:]) and (train_y.shape[1:] == self.shapes[1][1:])
        if self.idx < self.config["iters"]:
            for chunk_x, chunk_y in batch_parallel(train_x, train_y, self.config["chunk_size"]):
                log("Optimizer: Transferring data to computing device", LOG_LEVEL_VERBOSE)
                # Assign the chunk to the shared variable
                self.var_x.set_value(chunk_x, borrow=False)
                self.var_y.set_value(chunk_y, borrow=False)
                # Iterate through the chunk
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
                    if self.idx % self.config["print_freq"] == 0:
                        log("Training score at iteration %i: %s" % (self.idx, str(minibatch_avg_cost)), LOG_LEVEL_INFO)
                    if self.idx % self.config["save_freq"] == 0:
                        log("Saving intermediate model state", LOG_LEVEL_INFO)
                        self.graph.save(self.config["save_prefix"] + "_iter_" + str(self.idx) + ".zip")
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
    Apply online random augmentation.
    """
    def __init__(self, name, shapes, config, buffer_size=10):
        super(Transformer, self).__init__(name, shapes, config, buffer_size)
        self.mean = None

    def init(self):
        self.mean=np.load("train_mean.npy")

    def process(self):
        data = self.pull()
        # Return if no data is there
        if not data:
            return False
        # Unpack
        data, label = data
        # Do processing
        log("Transformer: Processing data", LOG_LEVEL_VERBOSE)
        i_h = 240
        i_w = 320

        d_h = 60
        d_w = 80
        start = time.time()

        # Random crops
        cy = rng.randint(data.shape[2] - i_h, size=1)
        cx = rng.randint(data.shape[3] - i_w, size=1)
        data = data[:, :, cy:cy+i_h, cx:cx+i_w]
        data = data.astype(np.float32)

        # Project image crop corner onto depth scales
        cy = int(float(cy) * (float(d_h)/float(i_h)))
        cx = int(float(cx) * (float(d_w)/float(i_w)))
        label = label[:, cy:cy+d_h, cx:cx+d_w]

        # Do elementwise operations
        for idx in range(data.shape[0]):
            # Subtract mean
            data[idx] = data[idx] - self.mean
            # Flip with probability 0.5
            p = rng.randint(2)
            if p > 0:
                data[idx] = data[idx, :, :, ::-1]
                label[idx] = label[idx, :, ::-1]

            # RGB we mult with a random value between 0.8 and 1.2
            r = rng.randint(80,121) / 100.
            g = rng.randint(80,121) / 100.
            b = rng.randint(80,121) / 100.
            data[idx, 0] = data[idx, 0] * r
            data[idx, 1] = data[idx, 1] * g
            data[idx, 2] = data[idx, 2] * b


        end = time.time()
        log("Transformer: Processing took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
        # Try to push into queue as long as thread should not terminate
        self.push((data, label))
        return True



