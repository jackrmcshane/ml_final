import os
import random
import numpy as np

import torch
import torch.nn
from torchtext import data





class DNN(nn.Module):

    def __init__(self):

        super(DNN, self).__init__()

        # define the architecture
        self.net = nn.Sequential(
                # 3 forward layers
                # 1 layer having 100 neurons
                # 2 layers having 50 neurons
                # output having 2; one for each class
                # relu activations

                # first layer
                nn.Linear(hyp.n_inputs, hyp.h1_neurons)
                nn.ReLU()

                # second layer
                nn.Linear(hyp.h1_neurons, hyp.h2_neurons)
                nn.ReLU()

                # third layer
                nn.Linear(hyp.h2_neurons, hyp.h3_neurons)
                nn.ReLU()

                # fourth layer
                # this will be the output layer
                # it will have two output neurons
                # one representing good, one representing bad
                # softmax will be handled by the loss function: CrossEntropyLoss
                nn.Linear(hyp.h3_neurons, hyp.n_classes)
        )


        # maybe define initializations?




    # defining the forward pass of the function
    def forward(self, tweet):
        return self.net(tweet.float()) # convert tweet to embedded version of tweet to float types as dense layers only deal in floats





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




def run_training_val(epochs, model, train_iterator, val_iterator, optimizer, criterion, model_type):
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

        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}%')



if __name__ == '__main__':

    # do stuff


    # defining hyperparams
    # this class holds various pieces of information that help define the model and its structure
    class hyp:
        # params that define network structure
        self.n_inputs = # we will see
        self.h1_neurons = 100
        self.h2_neurons = 50
        self.h3_neurons = 50
        # large dimensionality reduction between last hidden layer and output layer, may have to change/see how it affects
        self.n_classes = 2

        batch_size = 100
