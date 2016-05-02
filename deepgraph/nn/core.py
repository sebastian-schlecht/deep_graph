import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization

from deepgraph.node import Node
from deepgraph.conf import rng
from deepgraph.nn.init import normal, constant

__docformat__ = 'restructedtext en'


class Data(Node):
    """
    Create a node which holds a variable to feed in data to the compute graph.
    Typically used for training and label data.
    Can be reshaped using the reshape parameter
    """
    def __init__(self, graph, name, v_type, shape=None, config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param type: theano.variable
        :param shape: Tuple
        :param config: Dict
        :return: Node
        """
        super(Data, self).__init__(graph, name, config=config)
        self.input = v_type(name)
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
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: Name
        :param config: Dict
        :return: Node
        """
        super(Reshape, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Reshape, self).setup_defaults()
        self.conf_default("shape", None)

    def alloc(self):
        if self.conf("shape") is None:
            raise AssertionError("Reshape nodes need a valid tuple for param 'shape'.")
        if len(self.conf("shape")) == 0:
            raise AssertionError("Make sure shape is a tuple.")
        if len(self.inputs) > 1:
            raise AssertionError("Reshape nodes can only have exactly one input.")
        self.output_shape = self.conf("shape")

    def forward(self):
        in_ = self.inputs[0].expression
        self.expression = in_.reshape(self.conf("shape"))


class Softmax(Node):
    """
    Compute the softmax of the input. n_in and n_out speciy the input/output sizes respectively
    """
    def __init__(self, graph, name, inputs=[],config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(Softmax, self).__init__(graph, name, inputs=inputs, config=config)
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True

    def setup_defaults(self):
        super(Softmax, self).setup_defaults()
        self.conf_default("out", None)
        self.conf_default("weight_filler", constant(1))
        self.conf_default("bias_filler", constant(1))

    def alloc(self):
        if self.conf("out") is None:
            raise AssertionError("Configuration parameter 'out' has to be specified.")
        if len(self.inputs) != 1:
            raise ValueError("Softmax nodes can only have one input. This node currently has " + str(len(self.inputs)))
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 2:
            raise AssertionError("Softmax nodes must have 2 dim input. Current input has " + str(len(in_shape)) + " inputs.")

        # For softmax dim 1 is number of samples, dim 2 is already the number of channels.
        n_in = in_shape[1]
        # Init weights
        if self.W is None:
            self.W = theano.shared(
                value=self.conf("weight_filler")((n_in, self.conf("out"))),
                name='W',
                borrow=True
            )
        if self.b is None:
            # Init bias
            self.b = theano.shared(
                value=self.conf("bias_filler")(self.conf("out")),
                name='b',
                borrow=True
            )
        # These are the params to be updated
        self.params = [self.W, self.b]
        # Remember to compute the output shape
        self.output_shape = (self.inputs[0].output_shape[0], self.conf("out"))

    def forward(self):
        # Setup the forward pass of this node
        # Since inputs holds an array of nodes, we use their expression attribute to compute the symbolic expression
        self.expression = T.nnet.softmax(T.dot(self.inputs[0].expression, self.W) + self.b)


class Argmax(Node):
    """
    Computes the argmax of the input. Typically follows a softmax node. Axis specifies the axis to compute the argmax along
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(Argmax, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Argmax, self).setup_defaults()
        self.conf_default("axis", 1)
        self.conf_default("keepdims", False)

    def alloc(self):
        if self.conf("keepdims"):
            self.output_shape = self.inputs[0].output_shape
        else:
            self.output_shape = tuple(x for i, x in enumerate(self.inputs[0].output_shape) if i != self.conf("keepdims"))

    def forward(self):
        if len(self.inputs) != 1:
            raise ValueError("Softmax nodes can only have one input. This node currently has " + str(len(self.inputs)))
        # Setup the forward pass of this node
        # Since inputs holds an array of nodes, we use their expression attribute to compute the symbolic expression
        self.expression = T.argmax(self.inputs[0].expression, axis=self.conf("axis"), keepdims=self.conf("keepdims"))


class Flatten(Node):
    """
    Flatten the input into a tensor with dimensions = dims
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param dims: Int
        :param is_output: Bool
        :return: Node
        """
        super(Flatten, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Flatten, self).setup_defaults()
        self.conf_default("dims", None)

    def alloc(self):
        if not isinstance(self.conf("dims"), int):
            raise AssertionError("Configuration property 'dims' is mandatory and has to be of type 'int'.")
        if self.conf("dims") < 2:
            raise AssertionError("The data pipeline currently needs 2 dimensions minimum.")
        inshape = self.inputs[0].output_shape
        self.output_shape = [inshape[i] for i in range(self.conf("dims"))]
        k = inshape[self.conf("dims")]
        for j in range(len(inshape) - (self.conf("dims")+1)):
            k *= inshape[self.conf("dims") + j + 1]
        self.output_shape[self.conf("dims") - 1] *= k
        self.output_shape = tuple(i for i in self.output_shape)

    def forward(self):
        if len(self.inputs) != 1:
            raise ValueError("Flatten nodes can only have one input. This node currently has " + str(len(self.inputs)))
        # Setup the forward pass of this node
        # Since inputs holds an array of nodes, we use their expression attribute to compute the symbolic expression
        self.expression = self.inputs[0].expression.flatten(self.conf("dims"))


class Dense(Node):
    """
    Implements a single fully connected node.
    """
    def __init__(self, graph, name, inputs=[],config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(Dense, self).__init__(graph, name, inputs=inputs, config=config)
        # Mandatory to be able to collect gradients
        self.computes_gradient = True

    def setup_defaults(self):
        super(Dense, self).setup_defaults()
        self.conf_default("out", None)
        self.conf_default("activation", T.tanh)
        self.conf_default("weight_filler", normal())
        self.conf_default("bias_filler", constant(0.1))

    def alloc(self):
        if len(self.inputs) != 1:
            raise AssertionError("Activation nodes must have exactly one input. Current layer has " + str(len(self.inputs)))
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 2:
            raise AssertionError("Fully connected nodes do not support input with more dimensions than 2 yet. Please flatten the input. first.")
        # We need the channel count to calculate how much neurons we need
        n_in = in_shape[1]

        if self.W is None:
            # Alloc mem for the weights
            W_values = self.conf("weight_filler")((n_in, self.conf("out")))
            if self.conf("activation") == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
            # Weights
            self.W = W
        # Bias
        if self.b is None:
            b_values = self.conf("bias_filler")(self.conf("out"))
            b = theano.shared(value=b_values, name='b', borrow=True)
            self.b = b
        # Parameters which should be updated during steps
        self.params = [self.W, self.b]
        # Out shape
        self.output_shape = (self.inputs[0].output_shape[0], self.conf("out"))

    def forward(self):
        if len(self.inputs) != 1:
            raise AssertionError("Activation nodes must have exactly one input. Current layer has " + str(len(self.inputs)))
        lin_output = T.dot(self.inputs[0].expression, self.W) + self.b
        self.expression = (
            lin_output if self.conf("activation") is None
            else self.conf("activation")(lin_output)
        )


class Concatenate(Node):
    """
    Concatenate two tensor along an axis
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: Name
        :param config: config
        :return:
        """
        super(Concatenate, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Concatenate, self).setup_defaults()
        self.conf_default("axis", 0)

    def alloc(self):
        if len(self.inputs) < 2:
            raise AssertionError("Concat nodes need more than one input.")
        for s in range(len(self.inputs[0].output_shape)):
            for i in range(1,len(self.inputs)):
                if (self.inputs[0].output_shape[s] != self.inputs[i].output_shape[s]) and s != self.conf("axis"):
                    raise AssertionError("Inputs have to be of the same dimension except for the axis to concatenate along.")
        # Compute new shape
        c_dim = 0
        for i in range(len(self.inputs)):
            c_dim += self.inputs[i].output_shape[self.conf("axis")]
        # Tuples are immutable, make an array instead and transform into a tuple afterwards
        new_shape = [s for s in self.inputs[0].output_shape]
        new_shape[self.conf("axis")] = c_dim
        self.output_shape = tuple(new_shape)

    def forward(self):
        expressions = [x.expression for x in self.inputs]
        self.expression = T.concatenate(expressions, axis=self.conf("axis"))


class Crop(Node):
    """
    Crop input along the last two axis. Used when cropping feature maps
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dicts
        :return: Node
        """
        super(Crop, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Crop, self).setup_defaults()
        self.conf_default("height",None)
        self.conf_default("width", None)
        self.conf_default("strategy", "center")

    def alloc(self):
        if self.conf("height") is None or self.conf("width") is None:
            raise AssertionError("Configuration height and width are mandatory for Crop nodes.")
        if len(self.inputs) != 1:
            raise AssertionError("Crop nodes can only have one input.")
        in_shape = self.inputs[0].output_shape
        if self.conf("height") > in_shape[2] or self.conf("height") > in_shape[3]:
            raise AssertionError("Runtime shape mismatch. Crop height/width is too small.")

        self.output_shape = (in_shape[0], in_shape[1], self.conf("height"), self.conf("width"))

    def forward(self):
        if self.conf("strategy") is "center":
            in_ = self.inputs[0].expression
            cy = (in_.shape[2] - self.conf("height")) / 2
            cx = (in_.shape[3] - self.conf("width")) / 2
            self.expression = in_[:, :, cy:cy+self.conf("height"), cx:cx+self.conf("width")]
        else:
            raise NotImplementedError()


class MSE(Node):
    """
    Compute the MSE for regression tasks
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return:
        """
        super(MSE, self).__init__(graph, name, inputs=inputs ,config=config)
        self.is_error = True

    def setup_defaults(self):
        super(MSE, self).setup_defaults()
        self.conf_default("root", False)

    def alloc(self):
        if len(self.inputs) != 2:
            raise AssertionError("MSE needs exactly two inputs to compute an error")
        self.output_shape = (1,)

    def forward(self):
        in_0 = self.inputs[0].expression
        in_1 = self.inputs[1].expression

        self.expression = T.mean((in_0 - in_1) ** 2)
        if self.conf("root"):
            self.expression = T.sqrt(self.expression)


class Error(Node):
    """
    Computes the mean error for classification tasks
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param is_output: Bool
        :return:
        """
        super(Error, self).__init__(graph, name, inputs=inputs, config=config)
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
    """
    Dropout node implementing stochastic dropout function.
    """
    # Statically keep track of dropout layers like a registry
    layers = []

    def __init__(self, graph, name, inputs=[], config={}):
        super(Dropout, self).__init__(graph, name, inputs=inputs, config=config)
        self.prob_drop = self.conf("prob")
        self.prob_keep = 1.0 - self.conf("prob")
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on
        self.mask = None

    def setup_defaults(self):
        super(Dropout, self).setup_defaults()
        self.conf_default("prob", 0.5)

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


class BN(Node):
    """
    Apply batch normalization to the tensor of activations
    """
    def __init__(self, graph, name, inputs=[],config={}):
        super(BN, self).__init__(graph, name, inputs=inputs, config=config)
        # Explicitly set the grad flag to false
        self.computes_gradient = False

        self.axes = None
        self.shape = None
        # Learned parameters - Stored for housekeeping
        self.gamma = None
        self.beta = None

    def setup_defaults(self):
        super(BN, self).setup_defaults()
        self.conf_default("mode", 'low_mem')
        self.conf_default("disable", False)
        self.conf_default("axes", "auto")
        self.conf_default("nonlinearity", None)

    def alloc(self):
        if len(self.inputs) != 1:
            raise AssertionError("BN nodes need exactly one input.")
        input_shape = self.inputs[0].output_shape
        # Compute shape
        self.axes = (0,)
        self.shape = [size for axis, size in enumerate(input_shape) if axis not in self.axes]
        if self.conf("axes") == 'auto':
            # default: normalize over all but the second axis
            self.axes = (0,) + tuple(range(2, len(input_shape)))
        # Allocate memory for this node
        if self.gamma is None:
            self.gamma = theano.shared(value=np.ones(self.shape, dtype=theano.config.floatX), name='gamma')
        if self.beta is None:
            self.beta = theano.shared(value=np.zeros(self.shape, dtype=theano.config.floatX), name='beta')
        self.params = [self.gamma, self.beta]

        # Set shape
        self.output_shape = input_shape

    def forward(self):
        input = self.inputs[0].expression
        # TODO Disable this switch once verified that BN is working
        if not self.conf("disable"):
            self.expression = batch_normalization(inputs=input,
                                                  gamma=self.gamma,
                                                  beta=self.beta,
                                                  mean=input.mean(axis=self.axes, keepdims=True),
                                                  std=input.std(axis=self.axes, keepdims=True)
                                                  )
        else:
            self.expression = input

        if self.conf("nonlinearity") is not None:
            self.expression = self.conf("nonlinearity")(self.expression)

    def set_params(self, params):
        """
        We need to override this function in order to allow BN nodes to be pickled correctly.
        :param params:
        :return:
        """
        if len(params) == 2:
            self.gamma = params[0]
            self.beta = params[1]
            self.params = [self.gamma, self.beta]
        elif len(params) == 1:
            raise AssertionError("Loading only on parameter for file - pickling must have failed.")


class Function(Node):
    """
    Define an arbitrary function inside the computational graph
    """
    def __init__(self, graph, name, inputs=[], config={}):
        super(Function, self).__init__(graph, name, inputs=inputs, config=config)

        # Explicitly set the grad flag to false
        self.computes_gradient = False

    def setup_defaults(self):
        super(Function, self).setup_defaults()
        self.conf_default("expression", None)
        self.conf_default("output_shape", None)

    def alloc(self):
        if len(self.inputs) != 1:
            raise AssertionError("BN nodes need exactly one input.")
        input_shape = self.inputs[0].output_shape
        if self.conf("output_shape") is None:
            self.output_shape = input_shape
        else:
            self.output_shape = self.conf("output_shape")

    def forward(self):
        inp = self.inputs[0].expression
        lambda_func = self.conf("expression")
        if lambda_func is None:
            raise AssertionError("Function Nodes need to have the 'expression' config set!")
        else:
            if not hasattr(lambda_func, '__call__'):
                raise AssertionError("'expression' config parameter has to be callable!")
            self.expression = lambda_func(inp)