import random

import torch
import torch.nn as nn

from tokens import tokens


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, dropout, teacher_forcing):
        super(Decoder, self).__init__()
        self.teacher_forcing = teacher_forcing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, ctx_vector, prev_hidden):
        output = self.embedding(ctx_vector).view(1, 1, -1)
        output = self.dropout(output)
        output, next_hidden = self.gru(output, prev_hidden)
        output = self.linear(output[0])
        output = self.softmax(output)
        return output, next_hidden


def train_decoder(model, ctx_vector, output, criterion, teacher_forcing, device):
    decoder_input = torch.tensor([[tokens['SOS']]]).to(device=device)
    decoder_hidden = ctx_vector
    loss = 0

    if teacher_forcing == 'curriculum':
        if random.random() < 0.5:
            for decoder_idx in range(output.size()[0]):
                decoder_output, decoder_hidden = model(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, output[decoder_idx])
                decoder_input = output[decoder_idx]
        else:
            for decoder_idx in range(output.size()[0]):
                decoder_output, decoder_hidden = model(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, output[decoder_idx])
                if decoder_input.item() == tokens['EOS']:
                    break
    return loss