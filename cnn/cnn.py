"""
Description:

cnn stuff
- this cnn architecture will consist of two convolutional layers
-- one with filter size 3
-- one with filter size 5

"""



import os
import torch
import torch.nn as nn



def CNN(nn.Module):

    def __init__(self, vocab_size, embed_size, n_filters, filter_sizes, pool_size, hidden_size, n_classes, dropout_rate):

        super(CNN, self).__init__()

        # the embedding layer
        # creates a lookup table where each row represents a word in a numerical format and converts the integer sequence into a dense vector representation
        # vocab_size: num unique words in the dictionary
        # embed_size: num dimensions to use for representing a single word
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList(
                [nn.conv1d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_size)) for fs in filter_sizes]
        )

        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)



    def forward(self, text, text_lengths):

        # recieves text dimensions in shape of [batch_size, sentence_len]
        # after passing the embedding layer, changes shape to [batch_size, sentence_len, embedding_dims]
        # unsqueeze: adds a dimension of len 1 to the tensor
        embedded = self.embedding(text).unsqueeze(1)
        convolution = [conv(embedded) for conv in self.convs]
        # squeeze: reduces dimensions of len 1 in a tensor
        max1 = self.max_pool1(convolution[0].squeeze())
        max2 = self.max_pool1(convolution[1].squeeze())
        # cat: concatenates two tensors along a given dimension
        cat = torch.cat((max1, max2), dim=2)
        # view flattens the tensor?
        # the function is used to flatten the tensor for each sample in the batch so that the output shape will be of the following form:
        # [batch_size, (n_filters)*(n_out1/pooling_window_size + n_out2/pooling_window_size)]
        x = cat.view(cat.shape[0], -1)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



if __name__ = '__main__':

    # stuff
    class hyp:
        lr = 1e-4
        barch_size = 100
        dropout_rate = .25
        embedding_size = 300 # ?
        max_doc_length = 100 # each sentence gets max 100 words
        val_size = .2
        max_vocab_size = 5000 # maximum vocabulary size
        seed = 1
        n_classes = 3
        hidden_size = 128
        pool_size = 2
        n_filters = 128
        filter_sizes = [3,8]
        n_epochs = 5



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'path'
    data_path = os.path.join(path, 'data')


    model_type = "cnn"
    data_type = 'token'
    tokenizer = lambda s: s.split() # word based

    text.build_vocab(train_data, max_size=hyp.max_size)
    label.build_vocab(train_data)
    vocab_size = len(text.vocab)

    train_iterator, val_iterator, test_iterator = create_iterator(train_data, val_data, test_data, batch_size, device)

    loss_func = nn.CrossEntropyLoss()
    cnn_model = CNN(vocab_size, hyp.embedding_size, hyp.n_filters, filter_sizes, pool_size, hyp.hidden_size, hyp.n_classes, hyp.dropout_rate)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=hyp.lr)
    run_train(num_epochs, cnn_model, train_iterator, val_iterator, optimizer, loss_func, model_type)
    cnn_model.load_state_dict(torch.load(os.path.join(path, 'saved_weights_CNN.pt')))
    test_loss, test_acc = evaluate(cnn_model, test_iterator, loss_func)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
