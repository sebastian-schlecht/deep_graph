import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.signal.pool import Pool as TPool, pool_2d
from theano.tensor.nnet.abstract_conv import bilinear_upsampling

from deepgraph.graph import Node
from deepgraph.nn.init import (normal, constant)

__docformat__ = 'restructedtext en'


class Conv2D(Node):
    """
    Combination of convolution and pooling for ConvNets
    """
    def __init__(self, graph, name, config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(Conv2D, self).__init__(graph, name, config=config)
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
            self.W = theano.shared(
                self.conf("weight_filler")(size=self.filter_shape),
                name='W_conv',
                borrow=True
            )

        if self.b is None:
            self.b = theano.shared(value=self.conf("bias_filler")(self.filter_shape[0]), name="b_conv", borrow=True)
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
            raise AssertionError("ConvPool layer can only have one input")

        # Use optimization in case the number of samples is constant during compilation
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

        # Build final expression
        if self.conf("activation") is None:
            self.expression = conv_out
        else:
            self.expression = self.conf("activation")(conv_out)


class Upsample(Node):
    """
    Upsample the input tensor along axis 2 and 3. The previous node has to provide 4D output
    """
    def __init__(self, graph, name, config={}):
        super(Upsample, self).__init__(graph, name, config=config)

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


class Pool(Node):
    """
    Downsample using the Theano pooling module
    """
    def __init__(self, graph, name, config={}):
        super(Pool, self).__init__(graph, name, config=config)

    def setup_defaults(self):
        super(Pool, self).setup_defaults()
        self.conf_default("kernel", (2, 2))
        self.conf_default("ignore_border", True)
        self.conf_default("stride", None)
        self.conf_default("padding", (0, 0))
        self.conf_default("mode", "max")

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
        self.expression = pool_2d(
            _in,
            self.conf("kernel"),
            self.conf("ignore_border"),
            self.conf("stride"),
            self.conf("padding"),
            self.conf("mode")
        )


class LRN(Node):
    """
    Local response normalization to reduce overfitting

    Original implementation from PyLearn 2.
    See https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py for details
    """
    def __init__(self, graph, name, config={}):
        """
        Constructor
        :param graph: Graph
        :param name: String
        :param config: Dict
        :return: Node
        """
        super(LRN, self).__init__(graph, name, config=config)
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


