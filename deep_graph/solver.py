import numpy as np

from deep_graph.constants import *


class Solver(object):
    """
    Solver class to optimize graph objects
    """
    def __init__(self, lr=0.1, momentum=0.9, weight_decay=0.0005):
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

    def optimize_with_var(self, n_epochs=20, test_freq=100, val_freq=100, print_freq=10):
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
        while epoch < n_epochs:
            epoch += 1
            for minibatch_index in range(self.graph.n_train_batches):
                idx = (epoch - 1) * self.graph.n_train_batches + minibatch_index
                # Train in any case
                minibatch_avg_cost = self.models[TRAIN](minibatch_index, self.learning_rate, self.momentum, self.weight_decay)
                # Print in case the freq is ok
                if idx % print_freq == 0:
                    print "Current training score ..."
                    print("\taverage is: " + str(minibatch_avg_cost))

                if VAL in self.models:
                    if idx % val_freq == 0:
                        print("Validating graph ...")
                        val_losses = np.array([self.models[VAL](i) for i in range(self.graph.n_val_batches)])
                        print("\taverage is: " + str(np.mean(val_losses, axis=0)))

    def optimize_without_var(self, n_epochs=20, test_freq=100, val_freq=100,train_input=None, val_input=None, test_input=None, batch_size=32, print_freq=10):
        """
        Not implemented yet. Should optimize a graph by iterating through the array. Please note that this copies the data to the GPU for each call (slow)
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
        raise NotImplementedError()

    def optimize(self, n_epochs=20, test_freq=100, val_freq=100, train_input=None,val_input=None, test_input=None, batch_size=None, print_freq=10):
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
        # Right now we only support two inputs for epoch optimization
        compiled_with_var = self.graph.compiled_with_var
        if not compiled_with_var:
            self.optimize_without_var(n_epochs, test_freq, val_freq, train_input, batch_size, print_freq)
        else:
            self.optimize_with_var(n_epochs, test_freq, val_freq, print_freq)









