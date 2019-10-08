# it's magic
# https://github.com/hongzimao/pensieve/issues/11
import os
TRAIN_ENTROPY = [5., 1., 0.5, 0.3, 0.1]
TRAIN_EPOCH = [10000, 10000, 20000, 10000, 10000]
for p, q in zip(TRAIN_ENTROPY, TRAIN_EPOCH):
    if p == 5.:
        os.system('python main-loop.py')
    else:
        os.system('python main-loop.py results/nn_model_ep_lastest.ckpt ' + str(p) + ' ' + str(q))
