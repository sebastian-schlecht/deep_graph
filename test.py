from deepgraph.graph import *
from deepgraph.solver import *
from deepgraph.nn.core import *
from deepgraph.nn.loss import *
from deepgraph.nn.conv import *
from deepgraph.utils.datasets import *

from theano.tensor.nnet import relu

data = load_data("data/mnist.pkl.gz")

train_x, train_y = data[0]
val_x, val_y = data[1]
batch_size = 600

g = Graph("test")
#g.load_weights("data/model.zip")

# Until now, sequence matters to map inputs to compiled model inlets
# In addition, we need to specify the shape we want to have the input in such that deepgraph
# can calculate necessary mem.
data = Data(g, "data", T.matrix, shape=(-1, 1, 28, 28))
label = Data(g, "label", T.ivector, shape=(-1,), config={
    "phase": PHASE_TRAIN
})
conv_1 = Conv2D(g, "conv_1", config={
    "n_channels": 20,
    "kernel_shape": (5, 5),
    "activation": relu
})
lrn_1 = LRN(g, "lrn_1")
pool_1 = Pool(g, "pool_1")
conv_2 = Conv2D(g, "conv_2", config={
    "n_channels": 50,
    "kernel_shape": (5, 5),
    "activation": relu,
})
lrn_2 = LRN(g, "lrn_2")
pool_2 = Pool(g, "pool_2")
flatten = Flatten(g, "flatten", config={
    "dims": 2
})
fc3 = Dense(g, "fc3", config={
    "n_out": 500,
    "activation": T.tanh
})
soft = Softmax(g, "soft", config={
    "n_out": 10
})
arg = ArgMax(g, "arg", config={
    "is_output": True
})

error           = Error(g, "error")
loss            = NegativeLogLikelyHoodLoss(g, "loss")
l1              = L2RegularizationLoss(g, "l1", config={
    "loss_weight": 0.001
})



data.connect(conv_1)
conv_1.connect(lrn_1)
lrn_1.connect(pool_1)
pool_1.connect(conv_2)
conv_2.connect(lrn_2)
lrn_2.connect(pool_2)
pool_2.connect(flatten)
flatten.connect(fc3)
fc3.connect(soft)
fc3.connect(l1)
label.connect(loss)
soft.connect(arg)
soft.connect(loss)
soft.connect(l1)
arg.connect(error)
label.connect(error)

g.compile(train_input=[train_x, train_y], val_input=[val_x, val_y], batch_size=batch_size)
log("Starting optimization phase", LOG_LEVEL_INFO)
solver = Solver(lr=0.1)
solver.load(g)
solver.optimize(100)
log("Saving model", LOG_LEVEL_INFO)
g.save("data/model.zip")

log("Testing inference", LOG_LEVEL_INFO)
for idx in range(20):
    i_train_x = train_x.get_value()[idx]
    print g.infer([i_train_x.reshape((1, 1, 28, 28))])


