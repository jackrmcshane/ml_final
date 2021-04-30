"""imports"""
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable as Var



"""define network structure"""
class DNN(nn.Module):

    def __init__(self):

        super(DNN, self).__init__()

        self.net = nn.Sequential(
                # first layer: 100 neurons
                nn.Linear(hyp.n_inputs, hyp.h1_neurons),
                nn.ReLU(),

                # second layer: 50 neurons
                nn.Linear(hyp.h1_neurons, hyp.h2_neurons),
                nn.ReLU(),

                # third layer
                nn.Linear(hyp.h2_neurons, hyp.h3_neurons),
                nn.ReLU(),

                # output layer
                # two output classes: one neg, one pos
                nn.Linear(hyp.h3_neurons, hyp.n_classes)
        )


    def forward(self, data):
        return self.net(data)






class Dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index,:]
        y = self.labels[index]
        return x,y



def get_uniq(df):
    uniq_words = set()
    for row in df.itertuples():
        uniq_words = set.union(uniq_words, row.tweet.split())
    return uniq_words


def bow_transform(uniq_words, df: pd.DataFrame):
    df = pd.DataFrame(0, index=df.index, columns=uniq_words)
    for i, row in df.iterrows():
        for word in row.tweet.split():
            if word in uniq_words:
                df.at[i, word] += 1



def create_iterators(train_set, val_set, test_set, batch_size):
    train_iterator = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iterator = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_iterator = data.DataLoader(test_set, batch_size=test_set.__len__(), shuffle=True)

    return train_iterator, val_iterator, test_iterator




"""define accuracy function"""
def accuracy(probs, target):
    predictions = probs.argmax(dim=1)
    n_correct = (predictions==target)
    accuracy = n_correct.sum().float() / float(target.size(0))
    return accuracy





def train(model, iterator, optimizer, lfunc):
    epoch_loss = 0
    epoch_acc = 0


    net.train(True)
    with torch.set_grad_enabled(True):
        for data, labels in iterator:
            opt.zero_grad() # zero out accumulated gradients

            data, labels = Var(data.float()), Var(labels.float())
            probs = model(data)

            loss = lfunc(probs, labels.long())
            acc = accuracy(probs, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)





def evaluate(model, iterator, lfunc):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.set_grad_enabled(False):
        for data, labels in iterator:

            data, labels = Var(data.float()), Var(lables.float())
            probs = model(data)

            loss = lfunc(probs, labels.long())
            acc = accuracy(probs, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def train_val(epochs, model, train_iterator, val_iterator, optimizer, lfunc):
    training_acc, training_loss = (list(), list())
    validation_acc, validation_loss = (list(), list())

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_iterator, optimizer, lfunc)
        val_loss, val_acc = evaluate(model, val_iterator, lfunc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_bow_dnn.pt')

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        validation_acc.append(val_acc)
        validation_loss.append(val_loss)

        print(f'Training Loss: {train_loss:.3f} | Training Acc.: {train_acc*100:.2f}%')
        print(f'Val. Loss: {val_loss:.3f} | Val. Acc.: {val_acc*100:.2f}%')

    return training_loss, training_acc, validation_loss, validation_acc





if __name__ == '__main__':

    """
    function headers:


    def get_uniq(df):
    def bow_transform(uniq_words, df: pd.DataFrame):
    def create_iterators(train_data, val_data, test_data, batch_size):
    def create_iterators(train_set, val_set, test_set, batch_size):
    def accuracy(probs, target):
    def train_val(epochs, model, train_iterator, val_iterator, optimizer, lfunc):
    """


    # load data
    datapath = '../data/'
    train_path = os.path.join(datapath, 'train_bin_labels.csv')
    #test_path = os.path.join(datapath, 'test_bin_labels.csv')

    #train_df = pd.read_csv(train_path, index_col=0)
    #test_df = pd.read_csv(test_path, index_col=0)
    df = pd.read_csv(train_path, index_col=0)


    # create proper splits
    #x = train_df.tweet
    #y = train_df.label
    #sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
    #for train_index, val_index in sss.split(x, y):
        #x_train = pd.DataFrame(x[train_index])
        #y_train = y[train_index]
        #x_val = pd.DataFrame(x[val_index])
        #y_val= y[val_index]


    # Create splits
    x = df.tweet
    y = df.label
    x_train_data = list()
    y_train_data = list()
    x_test_data = list()
    y_test_data = list()
    sss = StratifiedShuffleSplit(n_splits=2, test_size=.2)
    for train, test in sss.split(x, y):
        x_train_data.append(pd.DataFrame( x[train] ))
        x_test_data.append(pd.DataFrame( x[test] ))
        y_train_data.append(pd.DataFrame( y[train] ))
        y_test_data.append(pd.DataFrame( y[test] ))


    x_train = x_train_data[0]
    y_train = y_train_data[0]
    x_test = x_test_data[0]
    y_test = y_test_data[0]


    #for train, val in sss.split(x_train, y_train):
        #x_train = x_train[train]
        #x_val = x_train[val]
        #y_train = y_train[train]
        #y_val = y_train[val]
#
#
    #x_train = pd.DataFrame(x_train)
    #x_val = pd.DataFrame(x_val)
    #x_test = pd.DataFrame(x_test)


    # create bow representation
    uniq_words = get_uniq(x_train)
    print('num uniq words: ', len(uniq_words))

    train_data = bow_transform(uniq_words, x_train)
    val_data = bow_transform(uniq_words, x_val)
    test_data = bow_transform(pd.DataFrame(test_df.tweet))

    print(train_data.head())
    print('-----------------------------------------------------------------------------')
    print(val_data.head())
    print('-----------------------------------------------------------------------------')


    #have to add label columns back to the new bow dataframes


    # create dataloaders
    """
    train_iterator, val_iterator, test_iterator = create_iterators(train_data, val_data, test_data, hyp.batch_size)
    """
    # define hyperparams
    class hyp:
        #n_inputs =
        h1_neurons = 100
        h2_neurons = 50
        h3_neurons = 50
        n_classes = 2

        n_epochs = 50
        batch_size = 100
        lr = 1e-4

    # initialize network
    '''
    dnn = DNN()
    '''
    # initialize loss func
    '''
    lfunc = nn.CrossEntropyLoss()
    '''
    # initialize optimizer
    '''
    optim = optim.Adam(dnn.parameters(), lr=hyp.lr)
    '''
    # run training/validation
    '''
    train_loss, train_acc, val_loss, val_acc = train_val(hyp.n_epochs, dnn, train_iterator, val_iterator, optim, lfunc)
    '''
    # load best model
    '''
    dnn.load_state_dict(torch.load(os.path.join(path, 'saved_bow_dnn.pt')))
    '''
    # test model
    '''
    test_loss, test_acc = evaluate(dnn, test_iterator, lfunc)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    '''


    # export learning curves
    '''
    # accuracy plot
    fig, ax = plt.subplots(figsize=(15, 5))
    epochs = range(hyp.n_epochs)
    ax.set_title('Model Accuracy Across Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.xaxis.grid(True, ls='dotted')
    ax.yaxis.grid(True, ls='dotted')
    ax.plot(epochs, train_data, label='train', color='steelblue', marker='d', markersize=8, linestyle='dashdot', linewidth=2)
    ax.plot(epochs, val_data, label='val', color='coral', marker='d', markersize=8, linestyle='dashdot', linewidth=2)
    ax.legend()
    # export to file
    plt.savefig('accuracy.png')




    # loss plot
    fig, ax = plt.subplots(figsize=(15, 5))
    epochs = range(hyp.n_epochs)
    ax.set_title('Model Accuracy Across Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.xaxis.grid(True, ls='dotted')
    ax.yaxis.grid(True, ls='dotted')
    ax.plot(epochs, train_data, label='train', color='steelblue', marker='d', markersize=8, linestyle='dashdot', linewidth=2)
    ax.plot(epochs, val_data, label='val', color='coral', marker='d', markersize=8, linestyle='dashdot', linewidth=2)
    ax.legend()
    # export to file
    plt.savefig('loss.png')
    '''
