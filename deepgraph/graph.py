from __future__ import print_function

import numpy as np
import os
import theano
import theano.tensor as T
import theano.misc.pkl_utils as pkl_utils

from deepgraph.constants import *

__docformat__ = 'restructedtext en'


class Graph(object):
    """
    Graph class to manage computation nodes and compile them using theano
    """
    def __init__(self, name):
        """
        Constructor
        :param name: Graph name
        :return: Graph
        """
        #########################################
        # Graph meta
        #########################################
        self.name = name
        #########################################
        # Stored nodes
        #########################################
        self.node_names = {}
        self.nodes = []
        #########################################
        # Internal variables related to the training procedure
        #########################################
        self.cost = None
        self.index = T.lscalar('index')                 # index to a [mini]batch
        self.lr = T.scalar('learning_rate')            # learning rate to use for SGD
        self.momentum = T.scalar('momentum')            # momentum rate to use
        self.weight_decay = T.scalar('weight_decay')    # weight decay to use
        self.last_updates = []  # Array to buffer the last weight update for use with momentum
        #########################################
        # Model we use for training, testing and validation
        #########################################
        self.models = {}
        #########################################
        # Compilation meta
        #########################################
        self.compiled_with_var = False
        self.is_compiled = False
        self.n_train_batches = 0
        self.n_test_batches = 0
        self.n_val_batches = 0
        # Persisted parameter file
        self.data_store = None
        self.init_weights = False

    def add(self, node):
        """
        Add a node to the computation graph
        :param node:    Node
        :return:        None
        """
        if node.name in self.node_names:
            raise NameError("Nodes should have unique names per graph.")
        else:
            self.node_names[node.name] = 1
            self.nodes.append(node)
            node.parent = self

    def compile(self, train_inputs=None, test_inputs=None, val_inputs = None, batch_size=None):
        """
        Compile the graphs expression as a theano function. This method also computes gradients and weight update rules
        :param train_inputs: Theano Shared Variable (Optional)
        :param batch_size: Int (Optional)
        :return: None
        """
        #########################################
        # Pre-init weights for all nodes in case a dump has been loaded
        #########################################
        if self.init_weights:
            for node in self.nodes:
                if node.name in self.data_store:
                    node.set_params(self.data_store[node.name])
        #########################################
        # Init the forward path. Call init on all nodes which internally calls forward() to
        # construct the theano forward expressions
        #########################################
        node_list = []
        for node in self.nodes:
            if len(node.inputs) == 0:
                node_list.append(node)
        while len(node_list) != 0:
            next_list = []
            for node in node_list:
                # Call init on node to set it up
                node.init()
                # Next batch of nodes to process
                for o in node.outputs:
                    next_list.append(o)
            node_list = next_list
        #########################################
        # Collect cost information from all nodes
        #########################################
        costs = []
        params = []
        learning_rates = []
        outputs = []
        for node in self.nodes:
            # Collect cost
            if node.loss_weight > 0:
                costs.append((node.loss_weight, node.expression))
            if node.is_output:
                outputs.append(node.expression)
            # Collect parameters
            if node.computes_gradient and len(node.params) > 0:
                # Add the nodes parameters to the local list
                params += node.params
                # Add one entry of learning rate per parameter (needed later for zip)
                learning_rates += [node.lr if node.lr is not None else 1] * len(node.params)
        #########################################
        # Compute the global cost function with their respective weights
        #########################################
        idx = 0
        for weight, exp in costs:
            if idx == 0:
                self.cost = weight * exp
            else:
                self.cost += weight * exp
            idx += 1

        updates = self.sgd(params, learning_rates)
        #########################################
        # Collect inputs
        #########################################
        inputs = []
        for node in self.nodes:
            if node.is_data:
                inputs.append(node.input)
        #########################################
        # Either compile the data within this function if it has been provided or simply provide placeholders for it
        #########################################
        if train_inputs is not None:
            if batch_size is None:
                raise AssertionError("Batch size is needed when compiling the graph with input data.")
            self.n_train_batches = train_inputs[0].get_value(borrow=True).shape[0] // batch_size
            replacements = [(var[self.index * batch_size: (self.index + 1) * batch_size]) for var in train_inputs]
            # Zip them
            givens = zip(inputs, replacements)
            # Compile the function
            print("Invoking compiler ...")
            self.models[TRAIN] = theano.function(
                inputs=[self.index, self.lr, self.momentum, self.weight_decay],
                outputs=outputs,
                updates=updates,
                givens=givens
            )
            self.compiled_with_var = True
            #########################################
            # In case there are any val and test inputs we compile them here as well
            #########################################
            if test_inputs is not None:
                self.n_test_batches = test_inputs[0].get_value(borrow=True).shape[0] // batch_size
                replacements = [(var[self.index * batch_size: (self.index + 1) * batch_size]) for var in test_inputs]
                givens = zip(inputs, replacements)
                # Compile the function
                self.models[TEST] = theano.function(
                    inputs=[self.index, self.lr, self.momentum, self.weight_decay],
                    outputs=outputs,
                    updates=updates,
                    givens=givens
                )
            if val_inputs is not None:
                self.n_val_batches = val_inputs[0].get_value(borrow=True).shape[0] // batch_size
                replacements = [(var[self.index * batch_size: (self.index + 1) * batch_size]) for var in val_inputs]
                givens = zip(inputs, replacements)
                # Compile the function
                self.models[VAL] = theano.function(
                    inputs=[self.index],
                    outputs=outputs,
                    givens=givens
                )

        else:
            inputs += [self.lr, self.momentum, self.weight_decay]
            print("Invoking compiler ...")
            self.models[TRAIN] = theano.function(
                inputs=inputs,
                outputs=outputs,
                updates=updates
            )
            self.compiled_with_var = False

        self.is_compiled = True

    def sgd(self, params, learning_rates):
        """
        Compute update rules based on standard SGD formulae. Includes momentum and weight decay
        :param params: List
        :param learning_rates:  List
        :return: List
        """
        # Make space for old gradient updates
        for param in params:
            delta_before_i = theano.shared(value=np.zeros(param.get_value().shape, dtype=theano.config.floatX))
            self.last_updates.append(delta_before_i)
        # Construct gradient objects for each parameter
        if self.cost is None:
            raise AssertionError("At least one cost function is needed per graph in order to optimize it.")
        gparams = [T.grad(self.cost, param) for param in params]
        updates = []
        for param, grad, last_update, lr_scale in zip(params, gparams, self.last_updates, learning_rates):
            delta = - self.lr * lr_scale * grad + self.momentum * last_update - self.weight_decay * self.lr * param
            # Update each parameters per iteration
            updates.append((param, param + delta))
            # Also save the last weight update
            updates.append((last_update, delta))
        return updates

    def save(self, filename):
        """
        Save a graph to a file
        :param filename: The name of the file to save
        :return: Bool (success)
        """
        data_store = {}
        for node in self.nodes:
            name = node.name
            data_store[name] = node.params
        with open(filename, "wb") as f:
            pkl_utils.dump(data_store, f)

        return True

    def load_weights(self, filename):
        """
        Load weights from a pickled zip file and store it internally in a hash
        :param filename: String
        :return:
        """
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                self.data_store = pkl_utils.load(f)
                self.init_weights = True
                print("Info:\tFinetuning from file " + filename)
        else:
            print("Warning:\tModel file not found.")


class Node(object):
    def __init__(self, graph, name, is_output=False ):
        if graph is None:
            raise ValueError("Nodes need a parent graph to be assigned to.")
        if name is None:
            raise ValueError("Nodes need a unique name.")

        # Name
        self.name = name
        self.is_output = is_output
        # Flag for producing loss
        self.loss_weight = 0
        # Flag for data. This flag has to be set to True if the node should act as an input
        self.is_data = False
        # Flag grad. In case this flag is set to True,
        # the graph will collect information about the parameters during compilation
        self.computes_gradient = False
        # Init flag
        self.is_init = False
        # Keep track of all nodes feeding this node and those which are fed by this node
        self.inputs = []
        self.outputs = []
        # Graph parent instance
        self.parent = None
        # Forward function as a theano expression
        self.expression = None
        # Potential weights
        self.W = None
        self.b = None
        self.params = []
        # Output shape
        self.output_shape = None
        # Add to graph
        graph.add(self)

    def init(self):
        """
        Called during graph compilation. Recursively initializes all preceeding nodes if not already done.
        This enables that all nodes have access to previous nodes' output shapes and expressions
        Returns
        -------

        """

        if not self.is_init:
            for i in self.inputs:
                if not i.is_init:
                    i.init()
            # Call setup
            self.alloc()
            self.forward()
            self.is_init = True

    def forward(self):
        """
        Called during graph initialization. forward() is responsible of building the theano expression
        like self.expression = function(input). It will be called AFTER alloc within the init() call
        Each node is initialized after its predecessors such that the previous nodes have already setup
        their output expressions and -shapes.
        :return: None
        """
        raise NotImplementedError()

    def alloc(self):
        """
        During alloc, the output shapes have to be computed and shared variables can be created. Called during init BEFORE
        forward(). Each node's alloc() function is
        :return: None
        """
        raise NotImplementedError()

    def get_params(self):
        """
        Get current params
        :return: List
        """
        return self.params

    def set_params(self, params):
        """
        Set new parameters. By default W is first, b is second
        :param params: List
        :return: None
        """
        if len(params) == 2:
            self.W = params[0]
            self.b = params[1]
            self.params = [self.W, self.b]
        elif len(params) == 1:
            raise AssertionError("No nodes have one parameter yet.")

    def connect(self, successor):
        """
        Connect two nodes with each other
        :param successor: Node
        :return:
        """
        if not isinstance(successor, Node):
            raise TypeError("Successor node has to be of a derivative of type 'Node'")
        self.outputs.append(successor)
        successor.inputs.append(self)