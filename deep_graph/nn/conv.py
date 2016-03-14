import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

from deep_graph.graph import Node
from deep_graph.conf import rng


class Conv2DPool(Node):
    """
    Combination of convolution and pooling for ConvNets
    """
    def __init__(self, graph, name, filter_shape, image_shape, pool_size=(2, 2), border_mode='valid', subsample=(1, 1), activation=T.tanh, lr=1, is_output=False):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param filter_shape: Tuple
        :param image_shape: Tuple
        :param pool_size:  Tuple
        :param border_mode: String or Tuple
        :param subsample: Int or Tuple
        :param activation: theano.elemwise
        :param lr: Float
        :param is_output: Bool
        :return: Node
        """
        super(Conv2DPool, self).__init__(graph, name, is_output=is_output)
        # Relative learning rate
        self.lr = lr
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True
        # Init weights
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        assert image_shape[1] == filter_shape[1]
        self.subsample = subsample
        self.border_mode = border_mode
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation = activation


    def alloc(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) //
                   np.prod(self.pool_size))
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
                dtype=theano.config.floatX
            ),
            name='W_conv',
            borrow=True
        )
        b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values,name="b_conv", borrow=True)
        # These are the params to be updated
        self.params = [self.W, self.b]
        ##############
        # Output shape
        ##############
        inshape = self.inputs[0].output_shape
        if len(self.image_shape) != 4:
            raise AssertionError("Input has to be 4D with shape (batchsize, channels, height, width)")

        # We start with output shapes for conv ops
        self.output_shape = get_conv_output_shape(image_shape=self.image_shape, kernel_shape=self.filter_shape, border_mode=self.border_mode, subsample=self.subsample)
        # But we also do pooling, keep that in mind
        self.output_shape = (self.output_shape[0], self.output_shape[1], self.output_shape[2] / self.pool_size[0], self.output_shape[3] / self.pool_size[1])

    def forward(self):
        if len(self.inputs) > 1:
            raise AssertionError("ConvPool layer can only have one input")
        conv_out = conv.conv2d(
            input=self.inputs[0].expression,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape,
            border_mode=self.border_mode,
            subsample=self.subsample
        )
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.pool_size,
            ignore_border=True
        )
        # Build final expression
        if self.activation is None:
            raise AssertionError("Conv/Pool nodes need an activation function.")
        self.expression = self.activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class LRN(Node):
    """
    Local response normalization to reduce overfitting

    Original implementation from PyLearn 2.
    See https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py for details
    """
    def __init__(self, graph, name, alpha=1e-4, k=2, beta=0.75, n=5, is_output=False):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param alpha: Float
        :param k: Int
        :param beta: Float
        :param n: Int
        :param is_output: Bool
        :return: Node
        """
        super(LRN, self).__init__(graph, name, is_output=is_output)
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now")

        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def alloc(self):
        self.output_shape = self.inputs[0].output_shape
        
    def forward(self):
        if len(self.inputs) != 1:
            raise AssertionError("LRN nodes can only have one input.")
        in_ = self.inputs[0].expression

        half = self.n // 2
        sq = T.sqr(in_)

        ch, r, c, b = in_.shape

        extra_channels = T.alloc(0., ch + 2*half, r, c, b)

        sq = T.set_subtensor(extra_channels[half:half+ch,:,:,:], sq)

        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[i:i+ch,:,:,:]

        scale = scale ** self.beta

        self.expression = in_ / scale


