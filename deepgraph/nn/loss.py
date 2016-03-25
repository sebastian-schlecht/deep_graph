import theano.tensor as T

from deepgraph.graph import Node
from deepgraph.constants import *

__docformat__ = 'restructedtext en'


class NegativeLogLikelyHoodLoss(Node):
    """
    Compute the negative log likelyhood loss of a given input
    Loss weight specifies to which degree the loss is considered during the update phases
    """
    def __init__(self, graph, name, loss_weight=1.0, is_output=True, phase=PHASE_TRAIN):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(NegativeLogLikelyHoodLoss, self).__init__(graph, name, is_output=is_output, phase=phase)
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
    def __init__(self, graph, name, loss_weight=0.001, is_output=True, phase=PHASE_TRAIN):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(L1RegularizationLoss, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.loss_weight = loss_weight

    def alloc(self):
        self.output_shape = (1,)

    def forward(self):
        if len(self.inputs) != 2:
            raise AssertionError("This node needs exactly two inputs to calculate loss")
        if not (self.inputs[0].W is not None and self.inputs[1].W is not None):
            raise AssertionError("L1 Regularization needs two nodes with weights as preceeding nodes")
        self.expression = (
            abs(self.inputs[0].W).sum() + abs(self.inputs[1].W).sum()
        )


class L2RegularizationLoss(Node):
    """
    L1 regularization node for adjacent fc layers
    """
    def __init__(self, graph, name, loss_weight=1, is_output=True, phase=PHASE_TRAIN):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(L2RegularizationLoss, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.loss_weight = loss_weight

    def alloc(self):
        self.output_shape = (1,)

    def forward(self):
        if len(self.inputs) != 2:
            raise AssertionError("This node needs exactly two inputs to calculate loss")
        if not (self.inputs[0].W is not None and self.inputs[1].W is not None):
            raise AssertionError("L2 Regularization needs two nodes with weights as preceeding nodes")
        self.expression = (
            (self.inputs[0].W ** 2).sum() + (self.inputs[1].W ** 2).sum()
        )


class LogarithmicScaleInvariantLoss(Node):
    """
    Compute log scale invariant error for depth prediction
    """
    def __init__(self, graph, name, lambda_factor=0.0, loss_weight=1.0, is_output=True, phase=PHASE_TRAIN):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param lambda_factor: Float
        :param loss_weight: Float
        :param is_output: Bool
        :return: Node
        """
        super(LogarithmicScaleInvariantLoss, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.loss_weight = loss_weight
        self.lambda_factor = lambda_factor

    def alloc(self):
            self.output_shape = (1,)

    def forward(self):
            if len(self.inputs) != 2:
                raise AssertionError("This node needs exactly two inputs to calculate loss.")
            # Define our forward function
            in_0 = self.inputs[0].expression
            in_1 = self.inputs[1].expression
            eps = 0.00001
            MAX = 1000000
            # TODO Eval T.clip() here. It should be less of a problem with last layer relu units though
            # TODO It may return during scale two though
            diff = T.log(T.clip(in_0, eps, MAX)) - T.log(T.clip(in_1, eps, MAX))
            self.expression = T.mean(diff**2) - ((self.lambda_factor / (in_0.shape[0]**2)) * (T.sum(diff)**2))


class EuclideanLoss(Node):
    """
    Computes the loss according to the mean euclidean distance of the input tensors
    Equivalent to mean squared error (MSE)
    """
    def __init__(self, graph, name, loss_weight=1.0, is_output=True, phase=PHASE_TRAIN):
            """
            Constructor
            :param graph:  Graph
            :param name: String
            :param loss_weight: Float
            :param is_output: Boolean
            :return:
            """
            super(EuclideanLoss, self).__init__(graph, name, is_output=is_output, phase=phase)
            self.loss_weight = loss_weight

    def alloc(self):
            self.output_shape = (1,)

    def forward(self):
            if len(self.inputs) != 2:
                raise AssertionError("This node needs exactly two inputs to calculate loss.")
            # Define our forward function
            in_0 = self.inputs[0].expression
            in_1 = self.inputs[1].expression

            self.expression = T.mean((in_0 - in_1) ** 2)