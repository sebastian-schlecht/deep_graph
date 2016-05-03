from deepgraph.constants import *
from deepgraph.utils.logging import *
from deepgraph.utils.common import ConfigMixin

__docformat__ = 'restructedtext en'


class Node(ConfigMixin):
    """
    Generic node class. Implements the new config object pattern
    """
    def __init__(self, graph, name, inputs=[], config={}):
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

        # Connect oneself to input nodes, if specified any
        if len(inputs) > 0:
            for in_node in inputs:
                self.connect(in_node)

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
            if self.output_shape is None:
                raise AssertionError("Node %s has not set an output shape. Make sure to assign self.output_shape in alloc()" % self.name)
            log("Node - %s has shape %s" % (self.name, str(self.output_shape)), LOG_LEVEL_INFO)
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


