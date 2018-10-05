#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')

from mnist_nets.train_mnist_nets import main as build_mnist_nets
from mnist_nets.postprocess_mnist_nets import main as postprocess_mnist_nets
from mnist_metamodel.mnist_metamodel import config as config_metamodel
from mnist_metamodel.mnist_metamodel import main as train_metamodel

# 1. Build MNIST-NET dataset of diverse MNIST classifiers
# Alternatively, download
# https://datasets.d2.mpi-inf.mpg.de/joon18iclr/MNIST-NET.tar.gz
# and untar in the ./cache/ folder

if True:
    build_mnist_nets()
    postprocess_mnist_nets()

# 2. Train metamodels. See below example configurations and annotations carefully before trying your own config.

example_no = 1

if example_no == 1:
    # kennen-o approach with 5000 training models and 100 queries with top-5 ranking outputs
    # under the Random (R) split.
    METHOD = 'm'  # Refers to kennen-o
    N_TRAIN = 5000  # Can be chosen in range [100,5000]
    N_EPOCH = 200  # Default number of epochs used in the paper
    N_QUERY = 100  # Can be chosen in range [1,1000]
    OUTPUT = 'ranking-5'  # ranking-k refers to top-k ranking output
    SPLIT = 'rand'
    SPLIT_TR = [1]  # Train on split 1
    SPLIT_TE = [0]  # Test on split 0
    GPU = None  # No GPU
elif example_no == 2:
    # kennen-i approach with 3000 training models
    # under the Extrapolation (E) split, with splitting attribute {#conv}.
    METHOD = 'i'  # Refers to kennen-i
    N_TRAIN = 3000
    N_EPOCH = 200
    N_QUERY = 1  # kennen-i always submits a single query
    OUTPUT = 'argmax'  # kennen-i only requires argmax output
    SPLIT = 'ex^net/n_conv'  # Extrapolation (E) split, the format is 'ex^{attr1}^{attr2}'
    # where attr1 and attr2 are the splitting attributes. For the full list of attributes,
    # see the bottom of this script.
    SPLIT_TR = [0, 1]  # Train on splits 0 and 1 (corresponds to #conv=2 or 3 - see bottom of page)
    SPLIT_TE = [2]  # Test on split 2 (corresponds to #conv=4)
    GPU = 1  # GPU ID
elif example_no == 3:
    # kennen-io approach with 100 training models and 100 queries with score outputs
    # under the Extrapolation (E) split, with splitting attribute {#conv,#fc}.
    METHOD = 'mi'  # Refers to kennen-io
    N_TRAIN = 100
    N_EPOCH = 400  # Default number of epochs for kennen-io
    N_QUERY = 100
    OUTPUT = 'score'
    SPLIT = 'ex^net/n_conv^net/n_fc'  # Possible to set multiple splitting attributes separated via '^'
    SPLIT_TR = [0, 1]  # Train on #conv=#fc=2 or 3
    SPLIT_TE = [2]  # Test on #conf=#fc=4
    GPU = 0  # GPU ID

co = config_metamodel(
    control=dict(
        method=METHOD,
        data=dict(
            name='dnet10000',
            subset=N_TRAIN,
            eval=1000,
        ),
        seed=0,
        i=dict(
            init='randval',
            clip=[0, 1],
            noise='U1',
            opt=dict(
                optimiser='SGD',
                lr=0.1,
                weight_decay=0.0,
                batch_size=10,
            ),
        ),
        m=dict(
            name='mlp_3_1000',
            opt=dict(
                optimiser='SGD',
                lr=1e-4,
                weight_decay=0.01,
                batch_size=100,
            ),
        ),
        opt=dict(
            epochs=N_EPOCH,
            sequence=['m', 200, 50, 50],
            # sequence=['m', 1, 1, 1],
        ),
        setup=dict(
            nquery=N_QUERY,
            qseed=0,
            target='all',
            outrep=OUTPUT,
            split=SPLIT,
            splitidtr=SPLIT_TR,
            splitidte=SPLIT_TE,
        ),
    ),
    conf=dict(
        exp_phase='mnist_metamodel',
        balanced_eval=True,
        test_batch_size=10,
        test_epoch=1,
        save=False,
        overridecache=True,
        mode='train',
        gpu=GPU,
    )
)
with co:
    train_metamodel(co)

# List of attributes and possible choices
{
    'net/act': ['relu', 'elu', 'prelu', 'tanh'],
    'net/drop': ['normal', 'none'],
    'net/pool': ['max_2', 'none'],
    'net/ks': [3, 5],
    'net/n_conv': [2, 3, 4],
    'net/n_fc': [2, 3, 4],

    'opt/optimiser': ['SGD', 'ADAM', 'RMSprop'],
    'opt/batch_size': [256, 128, 64],

    'data/subset': ['all', 'half_0', 'half_1', 'quarter_0', 'quarter_1', 'quarter_2', 'quarter_3'],

    'etc/data_size': ['all', 'half', 'quarter'],
    'etc/ens': [0, 1],
    'etc/n_param': range(14, 22, 1),
}
