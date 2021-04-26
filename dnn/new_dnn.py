"""
    Description:
"""



import torch
import random
import numpy as np
import matplotlib.plyplot as plt
import itertools
import sklearn.metrics as metrics

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as Var




# defining hyperparams for the network
class hyp:
    n_epochs = 50
    batch_size = 100
    lr = .05
    n_inputs = 5000 # length of the unique words vector
    fc1_nodes = 100
    fc2_nodes = 50
    fc3_nodes = 50
    out_nodes = 1




# defining the network
class DNN(nn.Module):

    # overwriting init function
    def __init__(self):
        super(DNN, self).__init__()

        # defining the architecture
        self.net = nn.Sequential(
            # first fully connected layer
            nn.Linear(n_inputs, fc1_nodes),
            nn.ReLU(),

            # second layer
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(),

            # third layer
            nn.Linear(fc2_nodes, fc3_nodes)
        )



        def forward(self, data):
            return self.net(data)





# defines the training procedure for the network
# input
# - net: the neural network for which training is to be done
# - lfunc: the loss function to be used for evaluating performance
# - opt: the optimizer
def train(net, lfunc, opt):
    ncorrect = 0 # correct predictions across an epoch
    nsamples = 0 # num tweets seen
    tot_loss = 0.0 # loss across the epoch

    net.train(True)
    with torch.set_grad_enabled(True):
        for batch, labels in train_gen:

            opt.zero_grad() # zero out accumulated gradients
            batch, labels = Var(batch.float()), Var(labels.float())
            out = net(batch)
            print(out)

            loss = lfunc(out, labels.long()) # returns avg loss for the batch
            tot_loss += loss * hyp.batch_size # add tot loss of batch to loss of epoch

            # backprop
            loss.backward() # gradient calc
            opt.step() # update weights

            selected_class = 1 if out.item() > .5 else 0
