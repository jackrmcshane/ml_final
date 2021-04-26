"""
Description:

This DNN is using a word embedding model
"""



import os
import re
import random
import torch
import torch.nn as nn
from torchtext import data



"""
DATA PREPROCESSING STEPS
1. preprocessing and tokenization
2. building the vocabulary
3. loading the embedding vectors
4. padding the text
5. batching the data
"""




"""Preprocessing and Tokenization"""
# this was foudn @ medium.com
# do not have to use, already have a preprocessing function
def cleanup_text(texts):
    cleaned_text = []
    for text in texts:
        # remove punctuation
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)
        cleaned_text.append(text)
    return cleaned_text




"""Defining the Network"""


class DNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)


    def forward(self, text, text_lengths):
        text = text.fload() # dense layers only deal with float datatype
        x = self.fc1(text)
        preds = self.fc2(x)
        return preds





# defining loss func
# if using binary labels/classes, can use BCELoss
loss_func = nn.CrossEntropyLoss()



def create_iterator(train_data, val_data, test_data, batch_size, device):
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=batch_size,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device=device
    )

    return train_iterator, val_iterator, test_iterator



# accuracy func for multiclass problems
def accuracy(probs, target):
    predictions = probs.argmax(dim=1)
    n_correct = (predictions == target)
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




def run_train(epochs, model, train_iterator, val_iterator, optimizer, criterion, model_type):
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

    device = torch.device('cuda', if torch.cuda.is_available() else 'cpu')
    path = 'general path'
    datapath = os.path.join(path, 'data')


    model_type = 'Linear'
    data_type = 'token' # or: 'morph'

    tokenizer = lambda s: s.split() # word based


    # defining hyperparameters
    class hyp:
        lr = 1e-4
        batch_size = 100
        dropout_rate = .15
        embedding_size = 300
        max_document_length = 100 # each sentence limited to 100 words
        val_size = .2
        max_vocab_size = 5000
        seed = 1
        num_classes = 3
        num_epochs = 50
        hidden_size = 100








    """From above"""
    # Field
    # defines a datatype as well as instructions for converting to Tensor
    # - preprocessing: can remove if text already cleaned or can add my own function
    # - sequential: whether a datatype represents sequential data, if false->no tokenization applied
    # - fix_length: set length for padding
    text = data.Field(preprocessing=cleanup_text, tokenize=tokenizer, batch_first=True, include_lengths=True, fix_length=max_document_length)
    label = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)


    # these should be in the order of your dataset/dataframe
    fields = [('text', text), ('labels', label)]





    # torchtext.data.TabularDataset is a torch data structure that specifically deals with tabular datasets such as .csv
    train_data, test_data = data.TabularDataset.splits(
            path = './data/',
            train='train.csv',
            test='test.csv',
            format='csv',
            fields=fields,
            skip_header=False
    )


    # visualizing the first row as it is held in the appropriate objects
    print(vars(train_data[0]))
    print(train_data[0].text)
    print(train_data[0].labels)


    # creating final training and val sets
    train_data, val_data = train_data.split(split_ratio=.8, random_state=random.seed(seed))



    """Building the Vocab"""
    # torchtext creates a dict of all the unique words and arranges them in decreasing order in accordance to their frequency
    # then a unique index is assigned to each word and torchtext keeps these in mapping functions:
    # - obj.vocab.stoi (string to index)
    # - obj.vocab.itos (index to string)

    text.build_vocab(train_data, val_data, max_size=5000)
    label.build_vocab(train_data)

    # vocab size/ # unique words
    vocab_size = len(text.vocab)
    print(vocab_size)


    """
    if want to use word vectors, can load the embedding layer using pretrained or can load from local folder as below
    but i think the above code trains its own vocab vector by the looks of the code in this comment



    from torchtext import vocab
    embeddings = vocab.Vectors('glove.840B.300d.txt', './local_vector_dir/')
    text.build_vocab(train_data, val_data, max_size=5000)
    label.build_vocab(train_data)
    """


    # sort_key: function that determines how to sort the data in the validation and test set
    # by setting sort_key to lambda x: len(x.text), torchtext will sort the samples by thier lengths
    # this setting is required when using the 'pack_padded_sequence' which we will be using later


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size = 100,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device=device
    )






    """end from above"""









    text.build_vocab(train_data, max_size=hyp.max_vocab_size)
    label.build_vocab(train_data)
    vocab_size = len(text.vocab)

    train_iterator, val_iterator, test_iterator = create_iterator(
            train_data, val_data, test_data, hyp.batch_size, device)

    loss_func = nn.CrossEntropyLoss()
    dnn = DNN(hyp.max_document_length, hyp.hidden_size, hyp.num_classes)
    optimizer = torch.optim.Adam(dnn.parameters(), lr=hyp.lr)

    run_train(hyp.num_epochs, dnn, train_iterator, val_iterator, optimizer, loss_func, model_type)



    """Testing the Model"""
    # load the weights
    dnn.load_state_dict(torch.load(os.path.join(path, 'saved_weights_dnn.pt')))

    test_loss, test_acc = evaluate(dnn, test_iterator, loss_func)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
