import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.signal.pool import Pool as TPool, pool_2d
from theano.sandbox.cuda import dnn

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
        super(Conv2D, self).__init__(graph, name, config)
        # Relative learning rate

        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True
        # Init weights
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        self.set_conf_default("subsample", (1, 1))
        self.set_conf_default("border_mode", 'valid')
        self.set_conf_default("n_channels", 1)
        self.set_conf_default("kernel_shape", (2, 2))
        self.set_conf_default("activation", T.tanh)
        self.set_conf_default("weight_filler", normal())
        self.set_conf_default("bias_filler", constant(1))
        self.set_conf_default("cudnn", False)

        # Those two guys are used for housekeeping
        self.image_shape = None
        self.filter_shape = None

    def alloc(self):
        # Compute filter shapes and image shapes
        if len(self.inputs) != 1:
            raise AssertionError("Conv nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        self.image_shape = in_shape
        self.filter_shape = (
                    self.conf("n_channels"),
                    self.image_shape[1],
                    self.conf("kernel_shape")[0],
                    self.conf("kernel_shape")[1]
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
        self.output_shape = get_conv_output_shape(
            image_shape=fake_shape,
            kernel_shape=self.filter_shape,
            border_mode=self.conf("border_mode"),
            subsample=self.conf("subsample"))
        # But we also do pooling, keep that in mind
        # When propagating data, we keep the n in (n,c,h,w) fixed to -1 to make theano
        # infer it during runtime
        self.output_shape = (in_shape[0], self.output_shape[1], self.output_shape[2], self.output_shape[3])

    def forward(self):
        if len(self.inputs) > 1:
            raise AssertionError("Conv2D nodes can only have one input.")

        # If there is cudnn, we use that
        if self.conf("cudnn") is True:
            conv_out = dnn.dnn_conv(img=self.inputs[0].expression,
                                    kerns=self.W,
                                    subsample=self.conf("subsample"),
                                    border_mode=self.conf("border_mode")
                                    )
        else:
            # Use optimization in case the number of samples is constant during compilation
            if self.image_shape[0] != -1:
                conv_out = conv2d(
                    input=self.inputs[0].expression,
                    input_shape=self.image_shape,
                    filters=self.W,
                    filter_shape=self.filter_shape,
                    border_mode=self.conf("border_mode"),
                    subsample=self.conf("subsample")
                )
            else:
                conv_out = conv2d(
                    input=self.inputs[0].expression,
                    filters=self.W,
                    filter_shape=self.filter_shape,
                    border_mode=self.conf("border_mode"),
                    subsample=self.conf("subsample")
                )
        # Build final expression
        if self.conf("activation") is None:
            # TODO Do we really need one?
            raise AssertionError("Conv/Pool nodes need an activation function.")
        self.expression = self.conf("activation")(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class Conv2DPool(Node):
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
        super(Conv2DPool, self).__init__(graph, name, config)
        # Tell the parent graph that we have gradients to compute
        self.computes_gradient = True
        # Init weights
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        self.set_conf_default("subsample", (1, 1))
        self.set_conf_default("n_channels", 1)
        self.set_conf_default("kernel_shape", (2, 2))
        self.set_conf_default("pool_size", (3, 3))
        self.set_conf_default("pool_stride", None)
        self.set_conf_default("ignore_border", True)
        self.set_conf_default("border_mode", "valid")
        self.set_conf_default("activation", T.tanh)
        self.set_conf_default("weight_filler", normal())
        self.set_conf_default("bias_filler", constant(0))

        # House-keeping
        self.filter_shape = None
        self.image_shape = None

    def alloc(self):
        # Compute filter shapes and image shapes
        if len(self.inputs) != 1:
            raise AssertionError("Conv nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        self.image_shape = in_shape
        self.filter_shape = (
                    self.conf("n_channels"),
                    self.image_shape[1],
                    self.conf("kernel_shape")[0],
                    self.conf("kernel_shape")[1]
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
        self.output_shape = get_conv_output_shape(
            image_shape=fake_shape,
            kernel_shape=self.filter_shape,
            border_mode=self.conf("border_mode"),
            subsample=self.conf("subsample")
        )
        # But we also do pooling, keep that in mind
        # When propagating data, we keep the n in (n,c,h,w) fixed to -1 to make theano
        # infer it during runtime
        # TODO That formula is wrong for alternating pooling modes!!
        self.output_shape = (
            in_shape[0],
            self.output_shape[1],
            self.output_shape[2] / self.conf("pool_size")[0],
            self.output_shape[3] / self.conf("pool_size")[1]
        )

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
                border_mode=self.conf("border_mode"),
                subsample=self.conf("subsample")
            )
        else:
            conv_out = conv2d(
                input=self.inputs[0].expression,
                filters=self.W,
                filter_shape=self.filter_shape,
                border_mode=self.conf("border_mode"),
                subsample=self.conf("subsample")
            )
        # downsample each feature map individually, using maxpooling
        pooled_out = pool_2d(
            input=conv_out,
            ds=self.conf("pool_size"),
            st=self.conf("pool_stride"),
            ignore_border=self.conf("ignore_border")
        )
        # Build final expression
        if self.conf("activation") is None:
            raise AssertionError("Conv/Pool nodes need an activation function.")
        self.expression = self.conf("activation")(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class Upsample(Node):
    """
    Upsample the input tensor along axis 2 and 3. The previous node has to provide 4D output
    """
    def __init__(self, graph, name, config={}):
        super(Upsample, self).__init__(graph, name, config)
        self.set_conf_default("kernel_size", (2, 2))

    def alloc(self):
        if len(self.inputs) > 1:
            raise AssertionError("Unpool nodes only support one input.")
        in_shape = self.inputs[0].output_shape
        if len(in_shape) != 4:
            raise AssertionError("Input has to be 4D.")
        if in_shape[3] == 0 or in_shape[4] == 0:
            raise AssertionError("Input shape is invalid.")
        self.output_shape = (
            in_shape[0],
            in_shape[1],
            in_shape[2] * self.conf("kernel_size")[0],
            in_shape[3] * self.conf("kernel_size")[1])

    def forward(self):
        _in = self.inputs[0].expression
        self.expression = _in.repeat(
            self.conf("kernel_size")[0],
            axis=2
        ).repeat(
            self.conf("kernel_size")[1],
            axis=3
        )


class Pool(Node):
    """
    Downsample using the Theano pooling module
    """
    def __init__(self, graph, name, config={}):
        super(Pool, self).__init__(graph, name, config)
        self.set_conf_default("kernel_size", (3, 3))
        self.set_conf_default("ignore_border", True)
        self.set_conf_default("stride", None)
        self.set_conf_default("padding", (0, 0))
        self.set_conf_default("mode", "max")
        self.set_conf_default("cudnn", False)

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
            self.conf("kernel_size"),
            self.conf("ignore_border"),
            self.conf("stride"),
            (0, 0)
        )

    def forward(self):
        _in = self.inputs[0].expression
        # In case we got cudnn we use that op
        if self.conf("cudnn") is False:
            self.expression = pool_2d(
                input=_in,
                ds=self.conf("kernel_size"),
                ignore_border=self.conf("ignore_border"),
                st=self.conf("stride"),
                padding=self.conf("padding"),
                mode=self.conf("mode")
            )
        else:
            self.expression = dnn.dnn_pool(
                img=_in,
                ws=self.conf("kernel_size"),
                stride=self.conf("stride"),
                pad=self.conf("padding")
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
        super(LRN, self).__init__(graph, name, config)

        self.set_conf_default("alpha", 0.0001)
        self.set_conf_default("k", 2)
        self.set_conf_default("beta", 0.75)
        self.set_conf_default("n", 5)

        if self.conf("n") % 2 == 0:
            raise NotImplementedError("Only works with odd n for now.")

    def alloc(self):
        self.output_shape = self.inputs[0].output_shape

    def forward(self):
        if len(self.inputs) != 1:
            raise AssertionError("LRN nodes can only have one input.")
        in_ = self.inputs[0].expression

        half = self.conf("n") // 2
        sq = T.sqr(in_)

        ch, r, c, b = in_.shape
        extra_channels = T.alloc(0., ch + 2*half, r, c, b)
        sq = T.set_subtensor(extra_channels[half:half+ch, :, :, :], sq)
        scale = self.conf("k")
        for i in xrange(self.conf("n")):
            scale += self.conf("alpha") * sq[i:i+ch, :, :, :]
        scale = scale ** self.conf("beta")
        self.expression = in_ / scale


