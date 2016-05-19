import sys, time, os
sys.path.append('..')

from deepgraph.graph import *
from deepgraph.solver import *
from deepgraph.nn.core import *
from deepgraph.nn.loss import *
from deepgraph.nn.conv import *
from deepgraph.nn.init import *
from deepgraph.utils.datasets import *

from theano.tensor.nnet import relu

data = load_data("../data/mnist.pkl.gz")

train_x, train_y = data[0]
val_x, val_y = data[1]
batch_size = 600

def build_graph():
    g = Graph("test")

    # Until now, sequence matters to map inputs to compiled model inlets
    # In addition, we need to specify the shape we want to have the input in such that deepgraph
    # can calculate necessary mem. Use -1 for dynamic batch sizes

    data = Data(g, "data", T.matrix, shape=(-1, 1, 28, 28))
    label = Data(g, "label", T.ivector, shape=(-1,), config={
        "phase": PHASE_TRAIN
    })
    conv_0 = Conv2D(g, "conv_0", inputs=[data],config={
        "channels": 20,
        "kernel": (5, 5),
        "activation": relu,
        "weight_filler": xavier(gain="relu"),
        "bias_filler": constant(0)
    })
    """
    Take the chance and test BN nodes
    """
    bn_0 = BN(g, "bn_0", inputs=[conv_0])
    pool_0 = Pool(g, "pool_0", inputs=[bn_0], config={
        "kernel": (2, 2)
    })
    lrn_0 = LRN(g, "lrn_0", inputs=[pool_0])
    """
    Weight sharing between conv_10 and conv_11. This doesn't make too much sense to add results up in the end but it's fine for 
    testing purposes
    """
    conv_10 = Conv2D(g, "conv_10", inputs=[lrn_0], config={
        "channels": 50,
        "kernel": (5, 5),
        "activation": relu,
        "weight_filler": xavier(gain="relu"),
        "bias_filler": constant(0)
    })
    conv_11 = Conv2D(g, "conv_11", inputs=[lrn_0], config={
        "channels": 50,
        "kernel": (5, 5),
        "activation": relu,
        "weight_filler": shared(conv_10, "W"), # Share weights with conv_10
        "bias_filler": constant(0)
    })
    """
    Also test the new elemwise node needed for residual nets
    """
    add_1 = Elemwise(g, "add_1", inputs=[conv_10, conv_11], config={
        "op": "add"
    })
    bn_1 = BN(g, "bn_1", inputs=[add_1])
    pool_1 = Pool(g, "pool_1", inputs=[bn_1], config={
        "kernel": (2, 2)
    })
    lrn_1 = LRN(g, "lrn_1", inputs=[pool_1])
    flatten = Flatten(g, "flatten", inputs=[lrn_1],config={
        "dims": 2
    })
    hidden_0 = Dense(g, "hidden_0", inputs=[flatten], config={
        "out": 500,
        "weight_filler": xavier(),
        "bias_filler": constant(0.0001),
        "activation": T.tanh
    })
    soft = Softmax(g, "softmax", inputs=[hidden_0],config={
        "out": 10,
        "weight_filler": xavier(),
        "bias_filler": constant(0.0001)
    })

    argm = Argmax(g, "argmax", inputs=[soft],config={
        "is_output": True
    })

    # Error and loss terms
    error = Error(g, "error", inputs=[argm, label])
    loss = NegativeLogLikelyHoodLoss(g, "loss", inputs=[soft, label])
    l2 = L2RegularizationLoss(g, "l2", inputs=[soft, hidden_0],config={"loss_weight": 0.0001})
    return g

# Compile and optimize
g = build_graph()
g.compile(train_inputs=[train_x, train_y], val_inputs=[val_x, val_y], batch_size=batch_size)
log("Starting optimization phase", LOG_LEVEL_INFO)
solver = Solver(lr=0.1)
solver.load(g)
# With BN it only needs a few epochs to converge
solver.optimize(4)
solver.learning_rate = 0.01
solver.optimize(4)

# Do some inference on MNIST train data
log("Testing inference", LOG_LEVEL_INFO)
ct = 40
for idx in range(ct):
    i_train_x = train_x.get_value()[idx]
    print g.infer([i_train_x.reshape((1, 1, 28, 28))])
    
# Save the graph locally
log("Testing store/load functionality", LOG_LEVEL_INFO)
g.save("tmp.zip")

# Delete all objects
del g
del solver

"""
Rebuild everything. Inference-only example
"""
g = build_graph()
g.load_weights("tmp.zip")
g.compile(phase=PHASE_INFER) # phase=PHASE_INFER tells the graph to only compile inference models without weight updates

log("Testing inference with loaded weights", LOG_LEVEL_INFO)
sum_times = 0.
for idx in range(ct):
    i_train_x = train_x.get_value()[idx]
    start = time.time()
    val =  g.infer([i_train_x.reshape((1, 1, 28, 28))])
    end = time.time()
    sum_times += end-start
    print val

os.remove("tmp.zip")    
log("Test finished. Average inference duration during loaded-weights phase: " + str(sum_times/ct), LOG_LEVEL_INFO)





