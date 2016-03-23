import numpy as np
import theano
import theano.tensor as T

from deepgraph.graph import Node
from deepgraph.conf import rng
from deepgraph.constants import *
from deepgraph.nn.init import normal, constant

__docformat__ = 'restructedtext en'


class Data(Node):
    """
    Create a node which holds a variable to feed in data to the compute graph.
    Typically used for training and label data.
    Can be reshaped using the reshape parameter
    """
    def __init__(self, graph, name, type, shape=None, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param type: theano.variable
        :param shape: Tuple
        :param is_output: Bool
        :return: Node
        """
        super(Data, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.input = type(name)
        self.is_data = True
        self.shape = shape

    def alloc(self):
        if self.shape is None:
            raise AssertionError("Please provide an input shape for this node.")
        self.output_shape = self.shape

    def forward(self):
        if len(self.inputs) != 0:
            raise ValueError("Data nodes cannot have any inputs. This node currently has " + str(len(self.inputs)))
        # Input nodes just pass their input to the preceeding node
        # Input should be a Theano variable
        self.expression = self.input.reshape(self.shape)


class Reshape(Node):
    """
    Reshapes the previous tensor
    """
    def __init__(self, graph, name, shape, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: Name
        :param shape: Tuple(Int)
        :param is_output: Bool
        :return: Node
        """
        super(Reshape, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.shape = shape

    def alloc(self):
        self.output_shape = self.shape

    def forward(self):
        if self.shape is None:
            raise AssertionError("Reshape nodes need a valid tuple for param 'shape'.")
        if len(self.shape) == 0:
            raise AssertionError("Make sure shape is a tuple.")
        if len(self.inputs) > 1:
            raise AssertionError("Reshape nodes can only have exactly one input.")

        in_ = self.inputs[0].expression

        self.expression = in_.reshape(self.shape)


class Softmax(Node):
    """
    Compute the softmax of the input. n_in and n_out speciy the input/output sizes respectively
    """
    def __init__(self, graph, name, n_out, lr=1, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param n_in: Int
        :param n_out: Int
        :param lr: Float
        :param is_output: Bool
        :return: Node
        """
        super(Softmax, self).__init__(graph, name, is_output=is_output, phase=phase)
        # Relative learning rate
        self.lr = lr
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True
        self.n_out = n_out
        self.n_in = 0

    def alloc(self):
        if len(self.inputs) != 1:
            raise ValueError("Softmax nodes can only have one input. This node currently has " + str(len(self.inputs)))

        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 2:
            raise AssertionError("Softmax nodes must have 2 dim input. Current input has " + str(len(in_shape)) + " inputs.")

        # For softmax dim 1 is number of samples, dim 2 is already the number of channels.
        # For higher dims flatten their output shape down to 2 dims
        self.n_in = in_shape[1]
        # Init weights
        if self.W is None:
            self.W = theano.shared(
                value=constant(0)((self.n_in,self.n_out)),
                name='W',
                borrow=True
            )
        if self.b is None:
            # Init bias
            self.b = theano.shared(
                value=constant(0)(self.n_out),
                name='b',
                borrow=True
            )
        # These are the params to be updated
        self.params = [self.W, self.b]
        # Remember to compute the output shape
        self.output_shape = (self.inputs[0].output_shape[0], self.n_out)

    def forward(self):
        if len(self.inputs) != 1:
            raise ValueError("Softmax nodes can only have one input. This node currently has " + str(len(self.inputs)))
        # Setup the forward pass of this node
        # Since inputs holds an array of nodes, we use their expression attribute to compute the symbolic expression
        self.expression = T.nnet.softmax(T.dot(self.inputs[0].expression, self.W) + self.b)


class ArgMax(Node):
    """
    Computes the argmax of the input. Typically follows a softmax node. Axis specifies the axis to compute the argmax along
    """
    def __init__(self, graph, name, axis=1, keepdims=False, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param axis: Int
        :param is_output: Bool
        :return: Node
        """
        super(ArgMax, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.axis = axis
        self.keepdims = keepdims

    def alloc(self):
        if self.keepdims:
            self.output_shape = self.inputs[0].output_shape
        else:
            self.output_shape = tuple(x for i, x in enumerate(self.inputs[0].output_shape) if i != self.axis)

    def forward(self):
        if len(self.inputs) != 1:
            raise ValueError("Softmax nodes can only have one input. This node currently has " + str(len(self.inputs)))
        # Setup the forward pass of this node
        # Since inputs holds an array of nodes, we use their expression attribute to compute the symbolic expression
        self.expression = T.argmax(self.inputs[0].expression, axis=self.axis, keepdims=self.keepdims)


class Flatten(Node):
    """
    Flatten the input into a tensor with dimensions = dims
    """
    def __init__(self, graph, name, dims, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param dims: Int
        :param is_output: Bool
        :return: Node
        """
        super(Flatten, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.dims = dims

    def alloc(self):
        if self.dims < 2:
            raise AssertionError("The data pipeline currently needs 2 dimensions minimum.")
        inshape = self.inputs[0].output_shape
        self.output_shape = [inshape[i] for i in range(self.dims)]
        k = inshape[self.dims]
        for j in range(len(inshape) - (self.dims+1)):
            k *= inshape[self.dims + j + 1]
        self.output_shape[self.dims - 1] *= k
        self.output_shape = tuple(i for i in self.output_shape)

    def forward(self):
        if len(self.inputs) != 1:
            raise ValueError("Flatten nodes can only have one input. This node currently has " + str(len(self.inputs)))
        # Setup the forward pass of this node
        # Since inputs holds an array of nodes, we use their expression attribute to compute the symbolic expression
        self.expression = self.inputs[0].expression.flatten(self.dims)




class FC(Node):
    """
    Implements a single fully connected node. Activations can be specified in the constructor
    """
    def __init__(self, graph, name, n_out, activation=T.tanh, lr=1, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param n_in: Int
        :param n_out: Int
        :param activation: theano.Elemwise
        :param W: theano.shared
        :param b: theano.shared
        :param lr: Float
        :param is_output: Bool
        :param phase: Int
        :return: Node
        """
        super(FC, self).__init__(graph, name, is_output=is_output, phase=phase)
        # Activation function
        self.activation = activation
        # Mandatory to be able to collect gradients
        self.computes_gradient = True
        # Relative learning rate
        self.lr = lr
        self.n_in = 0
        self.n_out = n_out

    def alloc(self):
        if len(self.inputs) != 1:
            raise AssertionError("Activation nodes must have exactly one input. Current layer has " + str(len(self.inputs)))
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 2:
            raise AssertionError("Fully connected nodes do not support input with more dimensions than 2 yet. Please flatten the input. first.")
        # We need the channel count to calculate how much neurons we need
        self.n_in = in_shape[1]

        if self.W is None:
            # Alloc mem for the weights
            W_values = normal()((self.n_in, self.n_out))
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
            # Weights
            self.W = W
        # Bias
        if self.b is None:
            b_values = constant(0)(self.n_out)
            b = theano.shared(value=b_values, name='b', borrow=True)
            self.b = b
        # Parameters which should be updated during steps
        self.params = [self.W, self.b]
        # Out shape
        self.output_shape = (self.inputs[0].output_shape[0], self.n_out)

    def forward(self):
        if len(self.inputs) != 1:
            raise AssertionError("Activation nodes must have exactly one input. Current layer has " + str(len(self.inputs)))
        lin_output = T.dot(self.inputs[0].expression, self.W) + self.b
        self.expression = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )



class Error(Node):
    """
    Computes the mean error for classification tasks
    """
    def __init__(self, graph, name, is_output=True, phase=PHASE_TRAIN):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param is_output: Bool
        :return:
        """
        super(Error, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.is_error = True

    def alloc(self):
        self.output_shape = (1,)

    def forward(self):
        # check if y has same dimension of y_pred
        if len(self.inputs) != 2:
            raise ValueError("This node needs exactly two inputs to calculate the error")

        if self.inputs[0].expression.ndim != self.inputs[1].expression.ndim:
            raise TypeError(
                "Inputs have to be of similar shape"
            )
        if self.inputs[0].is_data:
            label = self.inputs[0].expression
            pred = self.inputs[1].expression
        else:
            label = self.inputs[1].expression
            pred = self.inputs[0].expression
        # check if y is of the correct datatype
        if label.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            self.expression = T.mean(T.neq(pred, label))
        else:
            raise NotImplementedError()


class Dropout(Node):

    layers = []     # Statically keep track of dropout layers

    def __init__(self, graph, name, prob=0.5, is_output=False, phase=PHASE_ALL):
        super(Dropout, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.prob_drop = prob
        self.prob_keep = 1.0 - prob
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on
        self.mask = None

    def alloc(self):
        self.output_shape = self.inputs[0].output_shape

    def forward(self):
        if len(self.inputs) > 1:
            raise AssertionError("Dropoutlayers only support one input.")
        inp = self.inputs[0].expression
        seed_this = rng.randint(0, 2**31-1)
        mask_rng = T.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=inp.shape)
        Dropout.layers.append(self)

        self.expression = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * inp + \
            self.flag_off * self.prob_keep * inp

    @staticmethod
    def set_dp_on():
        for i in range(0, len(Dropout.layers)):
            Dropout.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def set_dp_off():
        for i in range(0, len(Dropout.layers)):
            Dropout.layers[i].flag_on.set_value(0.0)