import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layer):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layer = num_layer

        if torch.cuda.is_available():
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layer).cuda()
            self.linear = nn.Linear(self.hidden_size, self.output_size).cuda()
            self.softmax = nn.LogSoftmax(dim=1).cuda()
        else:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layer)
            self.linear = nn.Linear(self.hidden_size, self.output_size)
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden, cell):
        out, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        softmax_linear = self.linear(out)
        output = self.softmax(softmax_linear[-1])
        return output, hidden, cell

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size))
        return hidden, cell