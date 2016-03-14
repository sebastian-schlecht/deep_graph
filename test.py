from deep_graph.graph import *
from deep_graph.solver import *
from deep_graph.nn.core import *
from deep_graph.nn.loss import *
from deep_graph.nn.conv import *
from deep_graph.tools.datasets import *

from theano.tensor.nnet import relu

data = load_data("data/mnist.pkl.gz")

train_x, train_y = data[0]
val_x, val_y = data[1]
batch_size = 500

g = Graph("test")

# Until now, sequence matters to map inputs to compiled model inlets
data            = Data(g, "data", T.matrix, shape=(batch_size, 1, 28, 28))
label           = Data(g, "label", T.ivector, shape=(batch_size,))
# Comp nodes
conv_pool_0     = Conv2DPool(g, "conv_1", image_shape=(batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), activation=relu)
lrn_0           = LRN(g, "lrn_0")
conv_pool_1     = Conv2DPool(g, "conv_2", image_shape=(batch_size, 20, 12, 12), filter_shape=(50, 20, 5, 5), activation=relu)
lrn_1           = LRN(g, "lrn_1")
flatten         = Flatten(g, "flatten", dims=2)
hidden_0        = FC(g, "tanh_0", n_in=50*4*4, n_out=500)
softm           = Softmax(g, "softmax", n_in=500, n_out=10)
argm            = ArgMax(g, "argmax")
# Losses/Error
error           = Error(g, "error")
loss            = NegativeLogLikelyHoodLoss(g, "loss")
l1              = L2RegularizationLoss(g, "l1", loss_weight=0.001)

data.connect(conv_pool_0)
conv_pool_0.connect(lrn_0)
lrn_0.connect(conv_pool_1)
conv_pool_1.connect(lrn_1)
lrn_1.connect(flatten)
flatten.connect(hidden_0)
hidden_0.connect(softm)
hidden_0.connect(l1)
label.connect(loss)
softm.connect(argm)
softm.connect(loss)
softm.connect(l1)
argm.connect(error)
label.connect(error)


print "Compiling ..."
g.compile(train_inputs=[train_x, train_y], val_inputs=[val_x, val_y], batch_size=batch_size)
print "Optimizing ..."
solver = Solver(lr=0.1)
solver.load(g)
solver.optimize(1)

