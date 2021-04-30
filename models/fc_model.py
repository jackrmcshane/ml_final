import torch.nn as nn
import torch.nn.functional as F



class DNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super(DNN, self).__init__()
        self.net = nn.Sequential(
                # first layer
                nn.Linear(input_size, hidden_size, bias=True),
                nn.ReLU(),
                # second layer
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                # third layer
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                # output layer
                nn.Linear(hidden_size, num_classes)
                # nn.CrossEntropyLoss will be the loss func and performs softmax funciton so don't need here
        )



    def forward(self, tweets, tweets_lengths):
        # dense layers deal only in floats
        return self.net(tweets.float())
