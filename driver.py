import os
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing.preproc import get_files
from util.utils import create_iterator, validate, train_val, export_graph
from models.fc_model import DNN
from models.cnn import CNN


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '/home/asimov/sentiment_analysis'
    datapath = '/home/asimov/sentiment_analysis/data'

    # model params definitions
    model_type = 'CNN'
    tokenizer = lambda s: s.split() # word-based model

    lr = 1e-4 # will be using Adam optimizer to start
    batch_size = 100
    dropout_rate = .25
    embedding_size = 300
    max_doc_len = 100 # allowed 100 words per tweet
    train_size = .8
    max_vocab_size = 5000
    seed = 1
    n_classes = 2 # good/bad sentiment analysis


    # get data and types
    train_data, val_data, test_data, Text, Label = get_files(datapath, train_size, max_doc_len, seed, tokenizer)

    # build vocab
    Text.build_vocab(train_data, max_size=max_vocab_size)
    Label.build_vocab(train_data)
    vocab_size = len(Text.vocab)

    # get iterators
    train_iter, val_iter, test_iter = create_iterator(train_data, val_data, test_data, batch_size, device)
    # loss func
    lfunc = nn.CrossEntropyLoss()




    # for DNN model
    if model_type == 'DNN':

        n_epochs = 50
        hidden_size = 50

        dnn = DNN(max_doc_len, hidden_size, n_classes)
        optimizer = optim.Adam(dnn.parameters(), lr=lr)


        train_loss, train_acc, val_loss, val_acc = train_val(n_epochs, dnn, train_iter, val_iter, optimizer, lfunc, model_type)
        # export graphs
        export_graph('dnn_loss.png', n_epochs, train_loss, val_loss, 'Loss Across Epochs (DNN)', 'Loss')
        export_graph('dnn_acc.png', n_epochs, train_acc, val_acc, 'Network Accuracy Across Epochs (DNN)', 'Accuracy')


        dnn.load_state_dict(torch.load(os.path.join(path, 'DNN_saved_state.pt')))
        test_loss, test_acc = validate(dnn, test_iter, lfunc)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')



    # for CNN model
    if model_type == 'CNN':

        hidden_size = 128
        poos_size=2
        n_filters=128
        filter_sizes=[3,8]
        n_epochs = 30

        cnn = CNN(vocab_size, embedding_size, n_filters, filter_sizes, poos_size, hidden_size, n_classes, dropout_rate)
        optimizer = optim.Adam(cnn.parameters(), lr=lr)

        train_loss, train_acc, val_loss, val_acc = train_val(n_epochs, cnn, train_iter, val_iter, optimizer, lfunc, model_type)
        export_graph('cnn_loss.png', n_epochs, train_loss, val_loss, 'Loss Across Epochs (CNN)', 'Loss')
        export_graph('cnn_acc.png', n_epochs, train_acc, val_acc, 'Network Accuracy Across Epochs (CNN)', 'Accuracy')

        # test model
        cnn.load_state_dict(torch.load(os.path.join(path, 'CNN_saved_state.pt')))
        test_loss, test_acc = validate(cnn, test_iter, lfunc)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
