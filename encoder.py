import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_words):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_words = max_words
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, sentences, hidden):
        embeds = self.embedding(sentences).view(1, 1, -1)
        output, hidden = self.gru(embeds, hidden)
        return output, hidden


def encode_seq(model, seq, device):
    hidden = torch.zeros(1, 1, model.hidden_size).to(device=device)
    outputs = torch.zeros(model.max_words + 1, model.hidden_size)  # + 1 for EOS token

    for idx in range(seq.size()[0]):
        output, hidden = model(seq[idx], hidden)

        # Used for attention
        outputs[idx] = output[0, 0]

    return hidden
