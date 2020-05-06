import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        #input gate weights
        self.Wi = nn.Linear(hidden_size, hidden_size)
        self.Ui = nn.Linear(input_size, hidden_size)
        #Forget gate weights
        self.Wf = nn.Linear(hidden_size, hidden_size)
        self.Uf = nn.Linear(input_size, hidden_size)
        #output gate weights
        self.Wo = nn.Linear(hidden_size, hidden_size)
        self.Uo = nn.Linear(input_size, hidden_size)
        #cell weights
        self.Wc = nn.Linear(hidden_size,hidden_size)
        self.Uc = nn.Linear(input_size, hidden_size)

        #Fully connected layer for output
        self.V = nn.Linear(hidden_size, output_size, bias=True)
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    def initMemory(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self,input_data, hidden, memory):
        i = F.sigmoid(self.Wi(hidden) + self.Ui(input_data))
        f = F.sigmoid(self.Wf(hidden) + self.Uf(input_data))
        o = F.sigmoid(self.Wo(hidden) + self.Uo(input_data))
        c_tilde = F.tanh(self.Wc(hidden) + self.Uc(input_data))
        c = f * memory + i * c_tilde
        h = o * F.tanh(c)

        output = F.softmax(self.V(h))
        return output, h, c
