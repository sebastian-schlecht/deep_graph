from __future__ import print_function

import numpy as np
import os
import theano
import theano.tensor as T
import theano.misc.pkl_utils as pkl_utils

from deepgraph.constants import *
from deepgraph.utils.logging import *
from deepgraph.utils.common import ConfigMixin

__docformat__ = 'restructedtext en'

print(
    "\n\n",
    " _____                _____                 _\n",
    "|  _  \              |  __ \               | |\n",
    "| | | |___  ___ _ __ | |  \/_ __ __ _ _ __ | |__\n",
    "| | | / _ \/ _ \ '_ \| | __| '__/ _` | '_ \| '_ \\\n",
    "| |/ /  __/  __/ |_) | |_\ \ | | (_| | |_) | | | |\n",
    "|___/ \___|\___| .__/ \____/_|  \__,_| .__/|_| |_|\n",
    "               | |                   | |\n",
    "               |_|                   |_|\n\n"
)
print("Available on GitHub: https://github.com/sebastian-schlecht/deepgraph\n")


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
        self.lr = T.scalar('learning_rate', dtype=theano.config.floatX)            # learning rate to use for SGD
        self.momentum = T.scalar('momentum', dtype=theano.config.floatX)            # momentum rate to use
        self.weight_decay = T.scalar('weight_decay', dtype=theano.config.floatX)    # weight decay to use
        self.last_updates = []  # Array to buffer the last weight update for use with momentum
        #########################################
        # Model we use for training, testing and validation
        #########################################
        self.models = {}
        #########################################
        # Compilation meta
        #########################################
        self.compiled_with_var = False
        self.phase = PHASE_ALL
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

    def compile(self, train_inputs=None, test_inputs=None, val_inputs = None, batch_size=None, phase=PHASE_ALL):
        """
        Compile the graphs expression as a theano function. This method also computes gradients and weight update rules
        :param train_inputs: Theano Shared Variable (Optional)
        :param batch_size: Int (Optional)
        :return: None
        """
        log("Graph - Setting up graph", LOG_LEVEL_INFO)
        #########################################
        # Pre-init weights for all nodes in case a dump has been loaded
        #########################################
        if self.init_weights:
            for node in self.nodes:
                if node.name in self.data_store:
                    node.set_params(self.data_store[node.name])
        # Free some memory
        self.init_weights = False
        self.data_store = None
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
        outputs = {}
        for node in self.nodes:
            # Collect error
            if node.is_error is True:
                # outputs.append(node.expression)
                outputs[node.name] = node.expression
            # Collect cost
            if node.is_loss:
                costs.append((node.conf("loss_weight"), node.expression))
                # outputs.append(node.expression)
                outputs[node.name] = node.expression
            # Collect parameters
            if node.computes_gradient and len(node.params) > 0:
                # Add the nodes parameters to the local list
                params += node.params
                lr = node.conf("learning_rate")
                # Add one entry of learning rate per parameter (needed later for zip)
                learning_rates += [lr if lr is not None else 1.0] * len(node.params)
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
        log("Graph - Invoking Theano compiler", LOG_LEVEL_INFO)
        if phase is PHASE_ALL or phase is PHASE_TRAIN:
            if train_inputs is not None:
                if batch_size is None:
                    raise AssertionError("Batch size is needed when compiling the graph with input data.")
                self.n_train_batches = train_inputs[0].get_value(borrow=True).shape[0] // batch_size
                replacements = [(var[self.index * batch_size: (self.index + 1) * batch_size]) for var in train_inputs]
                # Zip them
                givens = zip(inputs, replacements)
                # Compile the function
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
                self.models[TRAIN] = theano.function(
                    inputs=inputs,
                    outputs=outputs,
                    updates=updates
                )
                self.compiled_with_var = False

        # In case we also need to build an inference model, compile an apropriate one
        if phase is PHASE_ALL or phase is PHASE_INFER:
            infer_in = []
            infer_out = {}
            for node in self.nodes:
                if node.conf("phase") == PHASE_ALL or node.conf("phase") == PHASE_INFER:
                    if node.conf("is_output"):
                        infer_out[node.name] = node.expression
                    elif node.is_data:
                        infer_in.append(node.expression)
            self.models[INFER] = theano.function(
                inputs=infer_in,
                outputs=infer_out,
            )

        # Set flag to true
        self.phase = phase
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
        try:
            with open(filename, "wb") as f:
                pkl_utils.dump(data_store, f)
                log("Graph - Model file saved as: %s" % filename, LOG_LEVEL_INFO)
        except IOError:
            log("Graph - Saving failed with error", LOG_LEVEL_ERROR)
            return False
        return True

    def load_weights(self, filename):
        """
        Load weights from a pickled zip file and store it internally in a hash
        :param filename: String
        :return: None
        """
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                self.data_store = pkl_utils.load(f)
                self.init_weights = True
                log("Graph - Loading parameters from file '%s'" % filename, LOG_LEVEL_INFO)
                if self.is_compiled:
                    log("Graph - Compilation has already taken place. Loaded weights won't "
                        "have any effect until the graph is recompiled", LOG_LEVEL_WARNING)
        else:
            log("Graph - Model not found: '%s" % filename, LOG_LEVEL_WARNING)

    def infer(self, arguments):
        """
        Execute the forward path of the inference model
        :param arguments: Model input parameters
        :return: List of return values
        """
        if not self.is_compiled:
            raise AssertionError("Cannot infer until the model is compiled")
        # Make sure there is something in memory
        assert self.models[INFER] is not None
        # Call
        return self.models[INFER](*arguments)


class Node(ConfigMixin):
    """
    Generic node class. Implements the new config object pattern
    """
    def __init__(self, graph, name, config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param is_output: Bool
        :param phase: Int
        :return: Node
        """
        # Call to super
        super(Node, self).__init__()
        self.make_configurable(config)
        if graph is None:
            raise ValueError("Nodes need a parent graph to be assigned to.")
        if name is None:
            raise ValueError("Nodes need a unique name.")
        # Name
        self.name = name
        """
        Flags. These flags are considered to be node specific and cannot be changed by configuration
        """
        # Error flag. Similar to output but considered only during train.
        self.is_error = False
        # Flag for data. This flag has to be set to True if the node should act as an input
        self.is_data = False
        # Flag grad. In case this flag is set to True,
        # the graph will collect information about the parameters during compilation
        self.computes_gradient = False
        # Init flag
        self.is_init = False
        # Loss flag
        self.is_loss = False
        """
        Graph housekeeping
        """
        # Keep track of all nodes feeding this node and those which are fed by this node
        self.inputs = []
        self.outputs = []
        # Graph parent instance
        self.parent = None
        """
        Computational attributes
        """
        # Forward function as a theano expression
        self.expression = None
        # Potential weights
        self.W = None
        self.b = None
        self.params = []
        # Output shape
        self.output_shape = None
        # Add to a parent graph
        graph.add(self)

    def init(self):
        """
        Called during graph compilation. Recursively initializes all preceeding nodes if not already done.
        This enables that all nodes have access to previous nodes' output shapes and expressions
        :return: None
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
        return successor

    def setup_defaults(self):
        self.conf_default("learning_rate", 1.0)
        self.conf_default("phase", PHASE_ALL)
        self.conf_default("loss_weight", 0)
        self.conf_default("is_output", False)


