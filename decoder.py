import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokens import Tokens


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_words, attention, dropout, teacher_forcing):
        super(Decoder, self).__init__()
        self.attention = attention
        if self.attention:
            self.att_layer = nn.Linear(hidden_size * 2, max_words)
            self.att_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.teacher_forcing = teacher_forcing
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, ctx_vector, prev_hidden, encoder_outs):
        emb = self.embedding(ctx_vector).view(1, 1, -1)
        out = self.dropout(emb).squeeze(0)

        if self.attention:
            attention = self.att_layer(torch.cat((out, prev_hidden[0]), 1))
            weights = F.softmax(attention, dim=1)
            batch_product = torch.bmm(weights.unsqueeze(0), encoder_outs.unsqueeze(0))
            combined = torch.cat((out, batch_product[0]), 1)
            out = self.att_concat(combined).unsqueeze(0)

        out, next_hidden = self.gru(out, prev_hidden)
        out = self.linear(out[0])
        out = self.softmax(out)
        return out, next_hidden


def train_decoder(model, ctx_vector, output, criterion, teacher_forcing, outputs, device):
    decoder_input = torch.tensor([[Tokens.SOS.value]]).to(device=device)

    # Init first hidden cell with context vector from encoder
    decoder_hidden = ctx_vector
    loss = 0

    use_ground_truth = random.random() < 0.5

    if teacher_forcing == 'curriculum':
        for decoder_idx in range(output.size()[0]):
            decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, outputs)
            loss += criterion(decoder_output, output[decoder_idx])
            if use_ground_truth:
                decoder_input = output[decoder_idx]
            else:
                _, max_idx = decoder_output.topk(1)
                decoder_input = max_idx.squeeze().detach()
                if decoder_input.item() == Tokens.EOS.value:
                    break
    return loss
