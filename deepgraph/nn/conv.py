import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.signal.pool import Pool as TPool, pool_2d

from deepgraph.graph import Node
from deepgraph.constants import *
from deepgraph.nn.init import (normal, constant)

__docformat__ = 'restructedtext en'


class Conv2D(Node):
    """
    Combination of convolution and pooling for ConvNets
    """
    def __init__(self, graph, name, n_channels, kernel_shape, border_mode='valid', subsample=(1, 1), activation=T.tanh, lr=1, is_output=False, phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param n_channels: Int
        :param kernel_shape: Tuple
        :param border_mode: String or Tuple
        :param subsample: Int or Tuple
        :param activation: theano.elemwise
        :param lr: Float
        :param is_output: Bool
        :return: Node
        """
        super(Conv2D, self).__init__(graph, name, is_output=is_output, phase=phase)
        # Relative learning rate
        self.lr = lr
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True
        # Init weights
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit

        self.subsample = subsample
        self.border_mode = border_mode
        self.n_channels = n_channels
        self.kernel_shape = kernel_shape
        self.filter_shape = None
        self.image_shape = None
        self.activation = activation

    def alloc(self):
        # Compute filter shapes and image shapes
        if len(self.inputs) != 1:
            raise AssertionError("Conv nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        self.image_shape = in_shape
        self.filter_shape = (
                    self.n_channels,
                    self.image_shape[1],
                    self.kernel_shape[0],
                    self.kernel_shape[1]
                            )
        assert self.image_shape[1] == self.filter_shape[1]
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        if self.W is None:
            self.W = theano.shared(
                normal()(size=self.filter_shape),
                name='W_conv',
                borrow=True
            )

        if self.b is None:
            self.b = theano.shared(value=constant(1)(self.filter_shape[0]), name="b_conv", borrow=True)
        # These are the params to be updated
        self.params = [self.W, self.b]
        ##############
        # Output shape
        ##############
        # We construct a fakeshape to be able to use the theano internal helper
        fake_shape = (1, self.image_shape[1], self.image_shape[2], self.image_shape[3])
        # We start with output shapes for conv ops
        self.output_shape = get_conv_output_shape(image_shape=fake_shape, kernel_shape=self.filter_shape, border_mode=self.border_mode, subsample=self.subsample)
        # But we also do pooling, keep that in mind
        # When propagating data, we keep the n in (n,c,h,w) fixed to -1 to make theano
        # infer it during runtime
        self.output_shape = (in_shape[0], self.output_shape[1], self.output_shape[2], self.output_shape[3])

    def forward(self):
        if len(self.inputs) > 1:
            raise AssertionError("ConvPool layer can only have one input")

        # Use optimization in case the number of samples is constant during compilation
        if self.image_shape[0] != -1:
            conv_out = conv2d(
                input=self.inputs[0].expression,
                input_shape=self.image_shape,
                filters=self.W,
                filter_shape=self.filter_shape,
                border_mode=self.border_mode,
                subsample=self.subsample
            ) + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            conv_out = conv2d(
                input=self.inputs[0].expression,
                filters=self.W,
                filter_shape=self.filter_shape,
                border_mode=self.border_mode,
                subsample=self.subsample
            ) + self.b.dimshuffle('x', 0, 'x', 'x')
        # Build final expression
        if self.activation is None:
            raise AssertionError("Conv/Pool nodes need an activation function.")
        self.expression = self.activation(conv_out )


class Conv2DPool(Node):
    """
    Combination of convolution and pooling for ConvNets
    """
    def __init__(self,
                 graph,
                 name,
                 n_channels,
                 kernel_shape,
                 pool_size=(2, 2),
                 pool_stride=None,
                 border_mode='valid',
                 subsample=(1, 1),
                 activation=T.tanh,
                 lr=1,
                 is_output=False,
                 phase=PHASE_ALL):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param n_channels: Int
        :param kernel_shape: Tuple
        :param pool_size:  Tuple
        :param border_mode: String or Tuple
        :param subsample: Int or Tuple
        :param activation: theano.elemwise
        :param lr: Float
        :param is_output: Bool
        :return: Node
        """
        super(Conv2DPool, self).__init__(graph, name, is_output=is_output, phase=phase)
        # Relative learning rate
        self.lr = lr
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True
        # Init weights
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit

        self.subsample = subsample
        self.border_mode = border_mode
        self.n_channels = n_channels
        self.kernel_shape = kernel_shape
        self.filter_shape = None
        self.image_shape = None
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.activation = activation

    def alloc(self):
        # Compute filter shapes and image shapes
        if len(self.inputs) != 1:
            raise AssertionError("Conv nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        self.image_shape = in_shape
        self.filter_shape = (
                    self.n_channels,
                    self.image_shape[1],
                    self.kernel_shape[0],
                    self.kernel_shape[1]
                            )

        assert self.image_shape[1] == self.filter_shape[1]
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        if self.W is None:
            self.W = theano.shared(
                normal()(size=self.filter_shape),
                name='W_conv',
                borrow=True
            )
        if self.b is None:
            self.b = theano.shared(value=constant(0)(self.filter_shape[0]), name="b_conv", borrow=True)
        # These are the params to be updated
        self.params = [self.W, self.b]
        ##############
        # Output shape
        ##############
        # We construct a fakeshape to be able to use the theano internal helper
        fake_shape = (1, self.image_shape[1], self.image_shape[2], self.image_shape[3])
        # We start with output shapes for conv ops
        self.output_shape = get_conv_output_shape(image_shape=fake_shape, kernel_shape=self.filter_shape, border_mode=self.border_mode, subsample=self.subsample)
        # But we also do pooling, keep that in mind
        # When propagating data, we keep the n in (n,c,h,w) fixed to -1 to make theano
        # infer it during runtime
        intermediate = (1, self.output_shape[1], self.output_shape[2], self.output_shape[3])
        # Include pooling
        self.output_shape = TPool.out_shape(
            intermediate,
            self.pool_size,
            True,
            self.pool_stride,
        )
        self.output_shape = (in_shape[0], self.output_shape[1], self.output_shape[2], self.output_shape[3])
        pass

    def forward(self):
        if len(self.inputs) > 1:
            raise AssertionError("ConvPool layer can only have one input.")

        # Use optimization in case the number of samples is constant during compilation
        if self.image_shape[0] != -1:
            conv_out = conv2d(
                input=self.inputs[0].expression,
                input_shape=self.image_shape,
                filters=self.W,
                filter_shape=self.filter_shape,
                border_mode=self.border_mode,
                subsample=self.subsample
            ) + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            conv_out = conv2d(
                input=self.inputs[0].expression,
                filters=self.W,
                filter_shape=self.filter_shape,
                border_mode=self.border_mode,
                subsample=self.subsample
            ) + self.b.dimshuffle('x', 0, 'x', 'x')
        # downsample each feature map individually, using maxpooling
        pooled_out = pool_2d(
            input=conv_out,
            ds=self.pool_size,
            st=self.pool_stride,
            ignore_border=True
        )
        # Build final expression
        if self.activation is None:
            raise AssertionError("Conv/Pool nodes need an activation function.")
        self.expression = self.activation(pooled_out)


class Upsample(Node):
    """
    Upsample the input tensor along axis 2 and 3. The previous node has to provide 4D output
    """
    def __init__(self, graph, name, kernel_size=(2, 2), is_output=False, phase=PHASE_ALL):
        super(Upsample, self).__init__(self, graph, name, is_output=is_output, phase=phase)
        self.kernel_size = kernel_size

    def alloc(self):
        if len(self.inputs) > 1:
            raise AssertionError("Unpool nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 4:
            raise AssertionError("Input has to be 4D.")
        if in_shape[3] == 0 or in_shape[4] == 0:
            raise AssertionError("Input shape is invalid.")
        self.output_shape = (in_shape[0], in_shape[1], in_shape[2] * self.kernel_size[0], in_shape[3] * self.kernel_size[1])

    def forward(self):
        _in = self.inputs[0].expression
        self.expression = _in.repeat(self.kernel_size[0], axis=2).repeat(self.kernel_size[1], axis=3)


class Pool(Node):
    """
    Downsample using the Theano pooling module
    """
    def __init__(self, graph, name, kernel_size=(2, 2), ignore_border= True, stride=None,padding=(0,0), mode='max', is_output=False, phase=PHASE_ALL):
        super(Pool, self).__init__(graph, name, is_output=is_output, phase=phase)
        self.kernel_size = kernel_size
        self.stride = stride
        self.ignore_border = ignore_border
        self.padding = padding
        self.mode = mode

    def alloc(self):
        if len(self.inputs) > 1:
            raise AssertionError("Pool nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 4:
            raise AssertionError("Input has to be 4D.")
        if in_shape[2] == 0 or in_shape[2] == 0:
            raise AssertionError("Input shape is invalid.")

        # Invoke theano internal function for shape computation
        self.output_shape = TPool.out_shape(
            in_shape,
            self.kernel_size,
            self.ignore_border,
            self.stride,
            self.padding
        )

    def forward(self):
        _in = self.inputs[0].expression
        self.expression = pool_2d(_in, self.kernel_size, self.ignore_border, self.stride, self.padding, self.mode)


class LRN(Node):
    """
    Local response normalization to reduce overfitting

    Original implementation from PyLearn 2.
    See https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py for details
    """
    def __init__(self, graph, name, alpha=1e-4, k=2, beta=0.75, n=5, is_output=False, phase=PHASE_ALL):
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
        super(LRN, self).__init__(graph, name, is_output=is_output, phase=phase)
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now.")

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

        b, ch, r, c = in_.shape

        extra_channels = T.alloc(0., b, ch + 2*half, r, c)

        sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)

        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[:,i:i+ch,:,:]

        scale = scale ** self.beta
        
        self.expression= in_ / scale


