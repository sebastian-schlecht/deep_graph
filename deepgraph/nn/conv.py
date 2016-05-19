import theano
import theano.tensor as T
from theano import config
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.signal.pool import Pool as TPool, pool_2d
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
from theano.sandbox.cuda import dnn

from deepgraph.node import Node, register_node
from deepgraph.nn.init import (normal, constant)
from deepgraph.utils.logging import log
from deepgraph.constants import *

__docformat__ = 'restructedtext en'


@register_node
class Conv2D(Node):
    """
    Combination of convolution and pooling for ConvNets
    """
    use_cudnn = False
    
    def __init__(self, graph, name, inputs=[],config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(Conv2D, self).__init__(graph, name, inputs=inputs, config=config)
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True

        # Used for housekeeping later
        self.filter_shape = None
        self.image_shape = None

    def setup_defaults(self):
        super(Conv2D, self).setup_defaults()
        self.conf_default("channels", None)
        self.conf_default("kernel", None)
        self.conf_default("border_mode", 'valid')
        self.conf_default("subsample", (1, 1))
        self.conf_default("activation", T.tanh)
        self.conf_default("weight_filler", normal())
        self.conf_default("bias_filler", constant(1))

    def alloc(self):
        # Compute filter shapes and image shapes
        if len(self.inputs) != 1:
            raise AssertionError("Conv nodes only support one input: %s." % self.name)
        in_shape = self.inputs[0].output_shape
        self.image_shape = in_shape
        self.filter_shape = (
                    self.conf("channels"),
                    self.image_shape[1],
                    self.conf("kernel")[0],
                    self.conf("kernel")[1]
                            )
        assert self.image_shape[1] == self.filter_shape[1]
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        if self.W is None:
            self.W = self.conf("weight_filler")(size=self.filter_shape, name='W_' + self.name)

        if self.b is None:
            self.b = self.conf("bias_filler")(size=self.filter_shape[0], name='b_' + self.name)
        # These are the params to be updated
        self.params = [self.W, self.b]
        ##############
        # Output shape
        ##############
        # We construct a fakeshape to be able to use the theano internal helper
        fake_shape = (1, self.image_shape[1], self.image_shape[2], self.image_shape[3])
        # We start with output shapes for conv ops
        self.output_shape = get_conv_output_shape(image_shape=fake_shape, kernel_shape=self.filter_shape, border_mode=self.conf("border_mode"), subsample=self.conf("subsample"))
        # When propagating data, we keep the n in (n,c,h,w) fixed to -1 to make theano
        # infer it during runtime
        self.output_shape = (in_shape[0], self.output_shape[1], self.output_shape[2], self.output_shape[3])

    def forward(self):
        if len(self.inputs) > 1:
            raise AssertionError("Conv node can only have one input")

        # Use optimization in case the number of samples is constant during compilation
        if not Conv2D.use_cudnn:
            if self.image_shape[0] != -1:
                conv_out = conv2d(
                    input=self.inputs[0].expression,
                    input_shape=self.image_shape,
                    filters=self.W,
                    filter_shape=self.filter_shape,
                    border_mode=self.conf("border_mode"),
                    subsample=self.conf("subsample")
                ) + self.b.dimshuffle('x', 0, 'x', 'x')
            else:
                conv_out = conv2d(
                    input=self.inputs[0].expression,
                    filters=self.W,
                    filter_shape=self.filter_shape,
                    border_mode=self.conf("border_mode"),
                    subsample=self.conf("subsample")
                ) + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            log("Conv2D - Using DNN CUDA Module", LOG_LEVEL_VERBOSE)
            conv_out = dnn.dnn_conv(img=self.inputs[0].expression,
                                    kerns=self.W,
                                    subsample=self.conf("subsample"),
                                    border_mode=self.conf("border_mode"),
                                    ) + self.b.dimshuffle('x', 0, 'x', 'x')
        # Build final expression
        if self.conf("activation") is None:
            self.expression = conv_out
        else:
            self.expression = self.conf("activation")(conv_out)


@register_node
class Upsample(Node):
    """
    Upsample the input tensor along axis 2 and 3. The previous node has to provide 4D output
    """
    def __init__(self, graph, name, inputs=[], config={}):
        super(Upsample, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Upsample, self).setup_defaults()
        self.conf_default("kernel", (2, 2))
        self.conf_default("ratio", 2)
        self.conf_default("mode", "constant")

    def alloc(self):
        if len(self.inputs) > 1:
            raise AssertionError("Unpool nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 4:
            raise AssertionError("Input has to be 4D.")
        if in_shape[2] == 0 or in_shape[3] == 0:
            raise AssertionError("Input shape is invalid.")
        self.output_shape = (in_shape[0], in_shape[1], in_shape[2] * self.conf("kernel")[0], in_shape[3] * self.conf("kernel")[1])

    def forward(self):
        _in = self.inputs[0].expression
        if self.conf("mode") is "constant":
            self.expression = _in.repeat(self.conf("kernel")[0], axis=2).repeat(self.conf("kernel")[1], axis=3)
        else:
            self.expression = bilinear_upsampling(input=_in, ratio=self.conf("ratio"))


@register_node
class Pool(Node):
    """
    Downsample using the Theano pooling module
    """
    use_cudnn = False
    
    def __init__(self, graph, name, inputs=[], config={}):
        super(Pool, self).__init__(graph, name, inputs=inputs, config=config)

    def setup_defaults(self):
        super(Pool, self).setup_defaults()
        self.conf_default("kernel", (2, 2))
        self.conf_default("ignore_border", True)
        self.conf_default("stride", None)
        self.conf_default("padding", (0, 0))
        self.conf_default("mode", "max"),

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
            self.conf("kernel"),
            self.conf("ignore_border"),
            self.conf("stride"),
            self.conf("padding")
        )
        self.output_shape = tuple(self.output_shape)

    def forward(self):
        _in = self.inputs[0].expression
        if not Pool.use_cudnn:
            self.expression = pool_2d(
                _in,
                self.conf("kernel"),
                self.conf("ignore_border"),
                self.conf("stride"),
                self.conf("padding"),
                self.conf("mode")
            )
        else:
            log("Pool - Using DNN CUDA Module", LOG_LEVEL_VERBOSE)
            pad = self.conf("padding") if self.conf("padding") is not None else (0,0)
            stride = self.conf("stride") if self.conf("stride") is not None else self.conf("kernel")
            self.expression = dnn.dnn_pool(_in,
                                           ws=self.conf("kernel"),
                                           stride=stride,
                                           mode=self.conf("mode"),
                                           pad=pad
                                           )


@register_node
class LRN(Node):
    """
    Local response normalization to reduce overfitting

    Original implementation from PyLearn 2.
    See https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py for details
    """
    def __init__(self, graph, name, inputs=[], config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(LRN, self).__init__(graph, name, inputs=inputs, config=config)
        if self.conf("n") % 2 == 0:
            raise NotImplementedError("Only works with odd n for now.")

    def setup_defaults(self):
        super(LRN, self).setup_defaults()
        self.conf_default("alpha", 1e-4)
        self.conf_default("k", 2)
        self.conf_default("beta", 0.75)
        self.conf_default("n", 5)

    def alloc(self):
        if len(self.inputs) != 1:
            raise AssertionError("LRN nodes can only have one input.")
        self.output_shape = self.inputs[0].output_shape
        
    def forward(self):

        in_ = self.inputs[0].expression

        half = self.conf("n") // 2

        sq = T.sqr(in_)

        b, ch, r, c = in_.shape

        extra_channels = T.alloc(0., b, ch + 2*half, r, c)

        sq = T.set_subtensor(extra_channels[:, half:half+ch, :, :], sq)

        scale = self.conf("k")

        for i in xrange(self.conf("n")):
            scale += self.conf("alpha") * sq[:, i:i+ch, :, :]

        scale = scale ** self.conf("beta")
        
        self.expression = in_ / scale


