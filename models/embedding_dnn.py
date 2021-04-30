import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.legacy import data





class DNN(nn.Module):

    def __init__(self):

        super(DNN, self).__init__()

        # define the architecture
        self.net = nn.Sequential(
                # first layer
                nn.Linear(hyp.n_inputs, hyp.h1_neurons),
                nn.ReLU(),

                # second layer
                nn.Linear(hyp.h1_neurons, hyp.h2_neurons),
                nn.ReLU(),

                # third layer
                nn.Linear(hyp.h2_neurons, hyp.h3_neurons),
                nn.ReLU(),

                # fourth layer
                # this will be the output layer
                # it will have two output neurons
                # one representing good, one representing bad
                # softmax will be handled by the loss function: CrossEntropyLoss
                nn.Linear(hyp.h3_neurons, hyp.n_classes)
        )




    # defining the forward pass of the function
    def forward(self, tweets):
        return self.net(tweets.float()) # convert tweet to embedded version of tweet to float types as dense layers only deal in floats





def create_iterator(train_data, val_data, test_data, batch_size, device):
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=hyp.batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device
    )

    return train_iterator, val_iterator, test_iterator



def accuracy(probs, target):
    predictions = probs.argmax(dim=1)
    n_correct = (predictions==target)
    accuracy = n_correct.sum().float() / float(target.size(0))
    return accuracy



def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(test, text_lengths)
        loss = criterion(predictions, batch.labels.squeeze())
        acc = accuracy(predictions, batch.labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator) , epoch_acc / len(iterator)




# validate/test the model
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.labels)
            acc = accuracy(predictions, batch.labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator) , epoch_acc / len(iterator)




# have to mod this function for producing the graphs
def run_training_val(epochs, model, train_iterator, val_iterator, optimizer, criterion, model_type):

    # create lists for train acc & loss for visualization of the network's learing
    training_acc, training_loss = (list(), list())
    # create lists for val acc & loss for visualization of the network's learing
    eval_acc, eval_loss = (list(), list())


    best_val_loss = float('inf')
    for epoch in range(epochs):
        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        #val model
        val_loss, val_acc = evaluate(model, val_iterator, criterion)
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_weights'+model_type+'.pt')

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        eval_acc.append(val_acc)
        eval_loss.append(val_loss)

        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}%')

    return training_acc, training_loss, eval_acc, eval_loss












if __name__ == '__main__':


    """
    1. Define network -- done
    2. define hyp
    3. load and prep data -- done-ish
    4. init model
    5. define
        loss func -- done
        optimizer -- done
        scheduler (optional)
    6. define
        train loss & acc lists -- done
        val loss & acc lists -- done
    7. train and validate model -- done-ish
    8. test model and report accuracy -- done-ish
    9. graph results of the learning


    TODO
    - mod the run_train func
    - change the init function for the network or change the way it is instantiated below
    - split the dataset into train and test and write to csv files
    - figure out how to save matlab plots to files
    - write the graphing functions
    """


    model_type = 'DNN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # defining hyperparams
    # this class holds various pieces of information that help define the model and its structure
    class hyp:
        # params that define network structure
        n_inputs = 5000
        h1_neurons = 100
        h2_neurons = 50
        h3_neurons = 50
        # large dimensionality reduction between last hidden layer and output layer, may have to change/see how it affects
        n_classes = 2

        batch_size = 100
        n_epochs = 50
        lr = 1e-4
        embedding_size = 300 # max embedding size per word
        max_document_length = 100 # each tweet limited to 100 words
        max_vocab_size = 5000
        seed = 1






    tokenizer = lambda s: s.split() # word based sentiment model
    # creating torchtext data types and instructions for converting to tensors
    # used by torchtext in loading the data from the csv files
    text = data.Field(tokenize=tokenizer, batch_first=True, include_lengths=True, fix_length=hyp.max_document_length)
    label = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
    fields = [('text', text), ('labels', label)]



    # reading data from the csv files
    datapath = '../data/'
    train_data, test_data = data.TabularDataset.splits(
            path=datapath,
            train='train.csv',
            test='test.csv',
            format='csv',
            fields=fields,
            skip_header=True
    )


    # splitting the training data into train and val sets
    train_data, val_data = train_data.split(split_ratio=.8, random_state=random.seed(hyp.seed))



    """Building the Vocab"""
    """how does this work? what does it accomplish"""
    # torchtext creates a dict of all the unique words and arranges them in decreasing order in accordance to their frequency
    # then a unique index is assigned to each word and torchtext keeps these in mapping functions:
    # - obj.vocab.stoi (string to index)
    # - obj.vocab.itos (index to string)
    # building the vocab for word embedding and later input to the system?
    text.build_vocab(train_data, val_data, max_size=hyp.max_vocab_size) # build vocab for tweets
    label.build_vocab(train_data) # build vocab for the labels
    vocab_size = len(text.vocab)
    """"""



    # creating dataset iterators from the datasets
    train_iterator, val_iterator, test_iterator = create_iterator(
            train_data, val_data, test_data, hyp.batch_size, device)



    # defining loss function
    loss_func = nn.CrossEntropyLoss()
    # init dnn model
    # will have to change the instantiation to match the def above
    dnn = DNN()
    # define optimizer
    optimizer = optim.Adam(dnn.parameters(), lr=hyp.lr)




    # perform the training
    # will have to modify this to return train and val stats
    train_acc, train_loss, val_acc, val_loss = run_training_val(hyp.n_epochs, dnn, train_iterator, val_iterator, optimizer, loss_func, model_type)



    """Testing the Model"""
    # load the weights
    dnn.load_state_dict(torch.load(os.path.join(path, 'saved_weights_dnn.pt')))

    test_loss, test_acc = evaluate(dnn, test_iterator, loss_func)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')



    """export learning curves"""

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
