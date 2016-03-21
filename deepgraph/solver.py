import math

import numpy as np
import theano
import theano.tensor as T

from deepgraph.constants import *
from deepgraph.utils.logging import *
from deepgraph.utils.common import batch_parallel
from deepgraph.nn.core import Dropout

__docformat__ = 'restructedtext en'


class Solver(object):
    """
    Solver class to optimize graph objects
    """
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0005):
        """
        Constructor
        :param lr: Float
        :param momentum: Float
        :param weight_decay: Float
        :return: Solver
        """
        self.index = 0
        self.learning_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.graph = None
        self.input = []
        self.models = {}

    def load(self, graph):
        """
        Load a graph and store a ref locally
        :param graph: Graph
        :return: None
        """
        if not graph.is_compiled:
            raise AssertionError("Graph is not compiled. Please call graph.compile() first in order to optimize its weights.")
        self.graph = graph
        self.models = graph.models
        self.index = 0

    def optimize_with_var(
            self,
            n_epochs=20,
            test_freq=100,
            val_freq=100,
            print_freq=10):
        """
        Optimize a graph with pre compiled data
        :param n_epochs: Int
        :param test_freq: Int
        :param val_freq: Int
        :param print_freq: Int
        :return: None
        """
        # Start looping
        epoch = 0
        log("Optimizing for %i epoch(s)" % n_epochs, LOG_LEVEL_INFO)
        while epoch < n_epochs:
            epoch += 1
            for minibatch_index in range(self.graph.n_train_batches):
                idx = (epoch - 1) * self.graph.n_train_batches + minibatch_index
                # Train in any case
                minibatch_avg_cost = self.models[TRAIN](minibatch_index, self.learning_rate, self.momentum, self.weight_decay)
                # Print in case the freq is ok
                if idx % print_freq == 0:
                    log("Training score at iteration %i: %s" % (idx, str(minibatch_avg_cost)), LOG_LEVEL_INFO)

                if VAL in self.models:
                    if idx % val_freq == 0:
                        val_losses = np.array([self.models[VAL](i) for i in range(self.graph.n_val_batches)])
                        log("Validation score at iteration %i: %s" % (idx, str(np.mean(val_losses, axis=0))), LOG_LEVEL_INFO)

    def optimize_without_var(
            self,
            n_epochs=20,
            test_freq=100,
            val_freq=100,
            train_input=None,
            val_input=None,
            test_input=None,
            batch_size=64,
            print_freq=10):
        """
        Optimize a graph by iterating through the array. Please note that this copies the data to the GPU for each call (slow)
        :param n_epochs: Int
        :param test_freq: Int
        :param val_freq: Int
        :param train_input: Int
        :param val_input: Int
        :param test_input: Int
        :param batch_size: Int
        :param print_freq: Int
        :return:
        """
        epoch = 0
        log("Optimizing for %i epoch(s) with one batch transfer per cycle" % n_epochs, LOG_LEVEL_INFO)
        assert train_input is not None
        train_x, train_y = train_input
        assert len(train_x) == len(train_y)
        idx = 0
        while epoch < n_epochs:
            epoch += 1
            minibatch_index = 0
            for minibatch_x, minibatch_y in batch_parallel(train_x, train_y, batch_size):
                idx += 1
                # Train in any case
                minibatch_avg_cost = self.models[TRAIN](
                    minibatch_x,
                    minibatch_y,
                    self.learning_rate,
                    self.momentum,
                    self.weight_decay
                )
                # Print in case the freq is ok
                if idx % print_freq == 0:
                    log("Training score at iteration %i: %s" % (idx, str(minibatch_avg_cost)), LOG_LEVEL_INFO)

                if VAL in self.models:
                    if idx % val_freq == 0:
                        val_losses = np.array([self.models[VAL](i) for i in range(self.graph.n_val_batches)])
                        log("Validation score at iteration %i: %s" % (idx, str(np.mean(val_losses, axis=0))), LOG_LEVEL_INFO)
                minibatch_index += 1

    def optimize(
            self,
            n_epochs=20,
            test_freq=100,
            val_freq=100,
            train_input=None,
            val_input=None,
            test_input=None,
            batch_size=None,
            print_freq=10):
        """
        Optimize the parameters of a graph
        :param n_epochs: Int
        :param test_freq: Int
        :param val_freq: Int
        :param train_input: List or None
        :param val_input: List or None
        :param test_input: List or None
        :param batch_size: Int
        :param print_freq: Int
        :return: None
        """
        # Toggle any dropouts. During optimizatino we want to leverage that
        Dropout.set_dp_off()
        # Right now we only support two inputs for epoch optimization
        compiled_with_var = self.graph.compiled_with_var
        if not compiled_with_var:
            self.optimize_without_var(n_epochs=n_epochs, test_freq=test_freq, val_freq=val_freq, train_input=train_input, batch_size=batch_size, print_freq=print_freq)
        else:
            self.optimize_with_var(n_epochs, test_freq, val_freq, print_freq)

    def compile_and_fit(self, graph, epochs=20, train_input=None, batch_size=64, superbatch_size=128, print_freq=20):
        """
        Compile the graph and optimize using larger chunks of data
        :param graph: Graph
        :param epochs: Int
        :param train_input: Tuple
        :param batch_size: Int
        :param superbatch_size: Int
        :param print_freq: Int
        :return:
        """
        self.graph = graph
        self.models = graph.models
        self.index = 0
        train_x, train_y = train_input
        assert len(train_x) == len(train_y)
        assert len(train_x) > superbatch_size
        assert batch_size < superbatch_size

        # Generate two variables of fixed size which hold our superbatch
        var_x = theano.shared(np.asarray(train_x[0: superbatch_size], dtype=theano.config.floatX), borrow=True)
        var_y = theano.shared(np.asarray(train_y[0: superbatch_size], dtype=theano.config.floatX), borrow=True)

        # Compile the graph
        graph.compile(train_inputs=[var_x, var_y], batch_size=batch_size)

        # Now we iterate in super-batches
        idx = 0
        epoch = 0
        while epoch < epochs:
            epoch += 1
            for super_x, super_y in batch_parallel(train_x, train_y, superbatch_size):

                # Assign the superbatch to the shared variable
                var_x.set_value(super_x, borrow=True)
                var_y.set_value(super_y, borrow=True)
                # Iterate through the super batch
                n_iters = int(math.ceil(len(super_x) / float(batch_size)))
                for minibatch_index in range(n_iters):
                    idx += 1
                    minibatch_avg_cost = self.models[TRAIN](
                        minibatch_index,
                        self.learning_rate,
                        self.momentum,
                        self.weight_decay
                    )
                    # Print in case the freq is ok
                    if idx % print_freq == 0:
                        log("Training score at iteration %i: %s" % (idx, str(minibatch_avg_cost)), LOG_LEVEL_INFO)











