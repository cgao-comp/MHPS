import torch
from model.Merger import *

class Decoder(nn.Module):
    def __init__(self, input_size, user_size, opt):
        super(Decoder, self).__init__()
        if opt.norm == True:
            self.decoder = Decoder2L(input_size, user_size, opt.dropout)
        else:
            self.decoder = Decoder1L(input_size, user_size, opt.dropout)

    def forward(self, outputs):
        return self.decoder(outputs)



class Decoder2L(nn.Module):
    def __init__(self, input_size, user_size, dropout=0.1):
        super(Decoder2L, self).__init__()

        self.linear2 = nn.Linear(input_size, input_size * 2)
        self.linear1 = nn.Linear(input_size * 2, user_size)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, outputs):

        return self.linear1(self.dropout(torch.tanh(self.linear2(outputs))))

class Decoder1L(nn.Module):
    def __init__(self, input_size, user_size, dropout = 0.1):
        super(Decoder1L, self).__init__()

        self.linear1 = nn.Linear(input_size, user_size)
        init.xavier_normal_(self.linear1.weight)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, outputs):
        return self.linear1(self.dropout(outputs))
