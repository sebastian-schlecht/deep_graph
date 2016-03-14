import theano.tensor as T

from deep_graph.graph import Node


class NegativeLogLikelyHoodLoss(Node):
    """
    Compute the negative log likelyhood loss of a given input
    Loss weight specifies to which degree the loss is considered during the update phases
    """
    def __init__(self, graph, name, loss_weight=1.0, is_output=True):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(NegativeLogLikelyHoodLoss, self).__init__(graph, name, is_output)
        self.loss_weight = loss_weight

    def alloc(self):
        self.output_shape = (1,)

    def forward(self):
        if len(self.inputs) != 2:
            raise AssertionError("This node needs exactly two inputs to calculate loss")
        # Get the label
        if self.inputs[0].is_data:
            label = self.inputs[0].expression
            pred = self.inputs[1].expression
        else:
            label = self.inputs[1].expression
            pred = self.inputs[0].expression
        # Define our forward function
        self.expression = -T.mean(T.log(pred)[T.arange(label.shape[0]), label])



class L1RegularizationLoss(Node):
    """
    L1 regularization node for adjacent fc layers
    """
    def __init__(self, graph, name, loss_weight=0.001, is_output=True):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(L1RegularizationLoss, self).__init__(graph, name, is_output=is_output)
        self.loss_weight = loss_weight

    def alloc(self):
        self.output_shape = (1,)

    def forward(self):
        if len(self.inputs) != 2:
            raise AssertionError("This node needs exactly two inputs to calculate loss")
        if not (self.inputs[0].W is not None and self.inputs[1].W is not None):
            raise AssertionError("L1 Regularization needs two nodes with weights as preceeding nodes")
        self.expression = (
            abs(self.inputs[0].W).sum()
            + abs(self.inputs[1].W).sum()
        )


class L2RegularizationLoss(Node):
    """
    L1 regularization node for adjacent fc layers
    """
    def __init__(self, graph, name, loss_weight=1, is_output=True):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(L2RegularizationLoss, self).__init__(graph, name, is_output=is_output)
        self.loss_weight = loss_weight

    def alloc(self):
        self.output_shape = (1,)

    def forward(self):
        if len(self.inputs) != 2:
            raise AssertionError("This node needs exactly two inputs to calculate loss")
        if not (self.inputs[0].W is not None and self.inputs[1].W is not None):
            raise AssertionError("L2 Regularization needs two nodes with weights as preceeding nodes")
        self.expression = (
            (self.inputs[0].W ** 2).sum()
            + (self.inputs[1].W ** 2).sum()
        )