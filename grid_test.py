import itertools
import subprocess
import os
import math

common_args = [
    'python',
    'train_bilinear.py',
    '--learning_rate=0.001',
    '--batch_size=52',
    '--style_embedding_dim=20',
    '--content_embedding_dim=20'
]

decomps = [
    'cp',
    'tt'
]

# the classic tensor would have 20 * 20 * 1024 = 409600 params
# so we would like to investigate anywhere up to this

param_denoms = [
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
]

def get_ranks(num_params, decomp):
    # tries to get ranks with the appropriate number of parameters
    # takes the ceiling
    # returns a single rank and the true number of params
    if decomp == 'cp':
        # we will have two 20 x r matrices and one 1024 x r
        # ie. p = 2 * 20r + 1024r
        # so r = p / 1064
        rank = int(math.ceil(num_params/1064))
        params = rank * 1064
        return rank, params
    if decomp == 'tt':
        # we assume r1 and r2 are the same
        # so we will have two 20 * r matrices
        # and an r x 1024 x r tensor
        # ie. p = 2 * 20r + 1024r^2
        # this guy has a root at the following point
        rank = math.sqrt(5696) * math.sqrt(p) - 40
        rank /= 2048
        rank = int(math.ceil(rank))
        params = 40 * rank + 1024 * (rank**2)
        return rank, params
    raise ValueError('I do not know {}'.format(decomp))
    


for decomp, factor in itertools.product(decomps, param_denoms):
    print('~~~~~~~~~~~~')
    rank, params = get_ranks(409600 // factor, decomp)
    print('{}-{}, rank {}'.format(decomp, params, rank))
    logdir = os.path.join(
        'logs/prelim', decomp, '{}_rank{}'.format(params, rank))
    new_args = [
        '--logdir='+logdir,
        '--decomposition='+decomp,
        '--rank1={}'.format(rank),
        '--rank2={}'.format(rank)
        ]
    subprocess.run(common_args + new_args,
                   check=True)





