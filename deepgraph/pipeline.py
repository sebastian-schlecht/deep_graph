import threading
import Queue
import time

import theano
import numpy as np
import h5py

from deepgraph.utils.logging import log
from deepgraph.utils.common import batch_parallel, ConfigMixin, shuffle_in_unison_inplace, pickle_dump
from deepgraph.constants import *
from deepgraph.conf import rng
from deepgraph.nn.core import Dropout


class Pipeline(ConfigMixin):
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
        super(Pipeline, self).__init__()
        self.make_configurable(config)
        self.processors = []
        self.last_item = None
        self.stop_evt = threading.Event()

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
        # Keep track of the number of cycles executed
        idx = 0
        # We assume the first processor in our list is the inlet
        if len(self.processors) == 0:
            raise AssertionError("At least one processor needed to make a pipeline run.")
        proc = self.processors[0]
        if proc is None:
            raise AssertionError("At least one processor needed to make a pipeline run.")
        log("Pipeline - Starting computation", LOG_LEVEL_INFO)
        while not self.stop_evt.is_set():
            try:
                proc.q.put(Packet(identifier=idx, phase=PHASE_TRAIN, num=2, data=None), block=True)
                idx += 1
                if (idx % self.conf("validation_frequency")) == 0:
                    proc.q.put(Packet(identifier=idx, phase=PHASE_VAL, num=2, data=None), block=True)
                if idx >= self.conf("cycles"):
                    break
            except Exception as e:
                self.stop()
                raise e
        log("Pipeline - All commands have been dispatched", LOG_LEVEL_INFO)
        proc.q.put(Packet(identifier=idx, phase=PHASE_END, num=2, data=None), block=True)

    def stop(self):
        """
        Stop running processors by signaling their stop events
        :return: None
        """
        self.stop_evt.set()
        for proc in self.processors:
            proc.stop.set()

    def signal(self, message):
        """
        Receive a signal from child processors
        :param message:
        :return:
        """
        if message == Pipeline.SIG_FINISHED:
            log("Pipeline - Complete signal received. Pipeline stopping.", LOG_LEVEL_INFO)
            self.stop()
        if message == Pipeline.SIG_ABORT:
            log("Pipeline - Abort signal received. Pipeline stopping.", LOG_LEVEL_WARNING)
            self.stop()

    def setup_defaults(self):
        self.conf_default("validation_frequency", 500)
        self.conf_default("cycles", 10000)


class Packet(object):
    """
    Packet container for generic data flowing through the pipeline
    """
    def __init__(self, identifier, phase, num, data):
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
        self.data = data


class Processor(threading.Thread, ConfigMixin):
    """
    Generic processor class which implements a stage in the pipeline. Each processor may have exactly
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
        self.make_configurable(config)
        self.top = None
        self.bottom = None
        self.name = name
        self.shapes = shapes
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

    def pull(self):
        """
        Pull data from previous processor
        :return:
        """
        has_data = False
        value = None
        while not self.stop.is_set():
            try:
                value = self.q.get(block=True, timeout=Processor.QUEUE_FULL_OR_EMPTY_TIMEOUT)
                has_data = True
                break
            except Queue.Empty:
                pass
        # Return if no data is there
        if not has_data:
            return None
        return value

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

    def setup_defaults(self):
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
        assert self.conf("db") is not None
        self.db_handle = h5py.File(self.conf("db"))
        self.data_field = self.db_handle[self.conf("key_data")]
        self.label_field = self.db_handle[self.conf("key_label")]
        assert self.data_field.shape[0] == self.label_field.shape[0]
        self.thresh = int(self.conf("split_value") * self.data_field.shape[0])

    def process(self):
        """
        Read elements from the database in packes specified by "chunk_size". Feed each packet to the top processor
        in case its entry queue is not full
        :return: Bool
        """
        if self.top is not None:
            packet = self.pull()
            if packet is None:
                return False
            log("H5DBLoader - Preloading chunk", LOG_LEVEL_VERBOSE)
            start = time.time()
            if packet.phase == PHASE_TRAIN:
                c_size = self.conf("chunk_size")
                upper = min(self.thresh, self.cursor + c_size)
                data = self.data_field[self.cursor:upper]
                label = self.label_field[self.cursor:upper]

                self.cursor += c_size
                # Reset cursor in case we exceeded the array ranges
                if self.cursor > self.thresh:
                    self.cursor = 0

            # Load entire validation data for now (one val cycle)
            elif packet.phase == PHASE_VAL:
                upper = self.data_field.shape[0]
                data = self.data_field[self.thresh:upper]
                label = self.label_field[self.thresh:upper]
            # End phase or unknown
            else:
                data, label = (None, None)

            end = time.time()
            log("H5DBLoader - Fetching took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
            self.push(Packet(identifier=packet.id,
                             phase=packet.phase,
                             num=2,
                             data=(data, label)))
            return True
        else:
            # No top node found, enter spin wait time
            return False

    def setup_defaults(self):
        super(H5DBLoader, self).setup_defaults()
        self.conf_default("db", None)
        self.conf_default("key_label", "label")
        self.conf_default("key_data", "data")
        self.conf_default("chunk_size", 320)
        self.conf_default("split_value", 0.9)


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
        self.train_x = None
        self.train_y = None

        self.val_x = None
        self.val_y = None
        # Iteration index
        self.idx = 0
        # Train losses
        self.losses = []

    def init(self):
        """
        Allocate some shared var's we use to store chunks in. After that, compile the graph so that gradients
        can be computed. We only do training here, no inference so compile in training mode only.
        :return: None
        """
        # Theano shared variables
        self.train_x = theano.shared(np.ones(self.shapes[0], dtype=theano.config.floatX), borrow=True)
        self.train_y = theano.shared(np.ones(self.shapes[1], dtype=theano.config.floatX), borrow=True)

        self.val_x = theano.shared(np.ones(self.shapes[0], dtype=theano.config.floatX), borrow=True)
        self.val_y = theano.shared(np.ones(self.shapes[1], dtype=theano.config.floatX), borrow=True)
        # Load weights if necessary
        if self.conf("weights") is not None:
            self.graph.load_weights(self.conf("weights"))
        # Compile
        self.graph.compile(
            train_inputs=[self.train_x, self.train_y],
            val_inputs=[self.val_x, self.val_y],
            batch_size=self.conf("batch_size"),
            phase=PHASE_TRAIN)
        log("Optimizer - Compilation finished", LOG_LEVEL_INFO)
        self.idx = 0

    def process(self):
        """
        Take an element out of the entry queue and iterate through it. Mark that an additional call to batch_parallel
        is made in case the chunk_size of this node differs from the producing node.
        :return: Bool (True)
        """
        packet = self.pull()
        # Return if no data is there
        if not packet:
            return False
        # Train phase
        if packet.phase == PHASE_TRAIN:
            train_x, train_y = packet.data
            start = time.time()
            assert (train_x.shape[1:] == self.shapes[0][1:]) and (train_y.shape[1:] == self.shapes[1][1:])
            for chunk_x, chunk_y in batch_parallel(train_x, train_y, self.conf("chunk_size")):
                log("Optimizer - Transferring data to computing device", LOG_LEVEL_VERBOSE)
                # Assign the chunk to the shared variable
                self.train_x.set_value(chunk_x, borrow=True)
                self.train_y.set_value(chunk_y, borrow=True)
                # Iterate through the chunk
                n_iters = len(chunk_x) // self.conf("batch_size")
                for minibatch_index in range(n_iters):
                    log("Optimizer - Computing gradients", LOG_LEVEL_VERBOSE)
                    Dropout.set_dp_on()
                    self.idx += 1
                    minibatch_avg_cost = self.graph.models[TRAIN](
                        minibatch_index,
                        self.conf("learning_rate"),
                        self.conf("momentum"),
                        self.conf("weight_decay")
                    )
                    # Save losses
                    self.losses.append(minibatch_avg_cost)
                    # Print in case the freq is ok
                    if self.idx % self.conf("print_freq") == 0:
                        log("Optimizer - Training score at iteration %i: %s" % (self.idx, str(minibatch_avg_cost)), LOG_LEVEL_INFO)
                    if self.idx % self.conf("save_freq") == 0:
                        log("Optimizer - Saving intermediate model state", LOG_LEVEL_INFO)
                        self.graph.save(self.conf("save_prefix") + "_iter_" + str(self.idx) + ".zip")
                        # Dump loss too
                        pickle_dump(self.losses, self.conf("save_prefix") + "_iter_" + str(self.idx) + "_loss.pkl")
                        # np_loss = np.array(self.losses)
                        # np.save(self.conf("save_prefix") + "_iter_" + str(self.idx) + "_loss.npy", np_loss)
            end = time.time()
            log("Optimizer - Computation took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
            # Return true, we don't want to enter spin waits. Just proceed with the next chunk or stop
            return True
        # Validation phase
        elif packet.phase == PHASE_VAL:
            # Make sure we've got validation functions
            assert VAL in self.graph.models and self.graph.models[VAL] is not None
            log("Optimizer - Entering validation cycle", LOG_LEVEL_VERBOSE)
            train_x, train_y = packet.data
            start = time.time()
            results = {}
            for chunk_x, chunk_y in batch_parallel(train_x, train_y, self.conf("chunk_size")):
                log("Optimizer - Transferring data to computing device", LOG_LEVEL_VERBOSE)
                # Assign the chunk to the shared variable
                self.val_x.set_value(chunk_x, borrow=True)
                self.val_y.set_value(chunk_y, borrow=True)
                # Iterate through the chunk
                n_iters = len(chunk_x) // self.conf("batch_size")

                for minibatch_index in range(n_iters):
                    log("Optimizer - Computing gradients", LOG_LEVEL_VERBOSE)
                    Dropout.set_dp_off()
                    minibatch_avg_cost = self.graph.models[VAL](
                        minibatch_index
                    )
                    for key in minibatch_avg_cost:
                        if key not in results:
                            results[key] = []
                        results[key].append(minibatch_avg_cost[key])
            # Compute mean values
            for key in results:
                val = np.array(results[key])
                results[key] = val.mean()
            end = time.time()
            log("Optimizer - Computation took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
            log("Optimizer - Mean loss values for validation at iteration " + str(self.idx) + " is: " + str(results), LOG_LEVEL_INFO)
            return True

        elif packet.phase == PHASE_END:
            self.pipeline.signal(Pipeline.SIG_FINISHED)
            return True

    def setup_defaults(self):
        super(Optimizer, self).setup_defaults()
        self.conf_default("batch_size", 64)
        self.conf_default("learning_rate", 0.001)
        self.conf_default("momentum", 0.9)
        self.conf_default("weight_decay", 0.0005)
        self.conf_default("print_freq", 50)
        self.conf_default("save_freq", 10000)
        self.conf_default("chunk_size", 320)
        self.conf_default("weights", None)
        self.conf_default("save_prefix", "model")
        self.conf_default("lr_policy", "constant")


class Transformer(Processor):
    """
    Apply online random augmentation.
    """
    def __init__(self, name, shapes, config, buffer_size=10):
        super(Transformer, self).__init__(name, shapes, config, buffer_size)
        self.mean = None

    def init(self):
        if self.conf("mean_file") is not None:
            self.mean = np.load(self.conf("mean_file"))
        else:
            log("Transformer - No mean file specified.", LOG_LEVEL_WARNING)

    def process(self):
        packet = self.pull()
        # Return if no data is there
        if not packet:
            return False
        # Unpack
        data, label = packet.data
        # Do processing
        log("Transformer - Processing data", LOG_LEVEL_VERBOSE)
        i_h = 452
        i_w = 452

        d_h = 260
        d_w = 260
        start = time.time()
        # Mean
        if packet.phase == PHASE_TRAIN or packet.phase == PHASE_VAL:
            data = data.astype(np.float32)
            if self.mean is not None:
                for idx in range(data.shape[0]):
                    # Subtract mean
                    data[idx] = data[idx] - self.mean.astype(np.float32)
            if self.conf("offset") is not None:
                label -= self.conf("offset")

        if packet.phase == PHASE_TRAIN:
            # Random crops
            cy = rng.randint(data.shape[2] - i_h, size=1)
            cx = rng.randint(data.shape[3] - i_w, size=1)
            data = data[:, :, cy:cy+i_h, cx:cx+i_w]

            # Project image crop corner onto depth scales
            cy = int(float(cy) * (float(d_h)/float(i_h)))
            cx = int(float(cx) * (float(d_w)/float(i_w)))
            label = label[:, cy:cy+d_h, cx:cx+d_w]

            # Do elementwise operations
            for idx in range(data.shape[0]):
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

            # Shuffle
            data, label = shuffle_in_unison_inplace(data, label)
        elif packet.phase == PHASE_VAL:
            # Center crop
            cy = (data.shape[2] - i_h) // 2
            cx = (data.shape[3] - i_w) // 2
            data = data[:, :, cy:cy+i_h, cx:cx+i_w]

            # Project image crop corner onto depth scales
            cy = int(float(cy) * (float(d_h)/float(i_h)))
            cx = int(float(cx) * (float(d_w)/float(i_w)))
            label = label[:, cy:cy+d_h, cx:cx+d_w]
        end = time.time()
        log("Transformer - Processing took " + str(end - start) + " seconds.", LOG_LEVEL_VERBOSE)
        # Try to push into queue as long as thread should not terminate
        self.push(Packet(identifier=packet.id, phase=packet.phase, num=2, data=(data, label)))
        return True

    def setup_defaults(self):
        super(Transformer, self).setup_defaults()
        self.conf_default("mean_file", None)
        self.conf_default("offset", None)


