import torch

from tokens import Tokens


def sentence_to_tensor(sentence, dict, device):
    ids = [dict.dict[w] if w in dict.dict else Tokens.UKN.value for w in sentence]
    ids.append(Tokens.EOS.value)
    return torch.tensor(ids, dtype=torch.long).view(-1, 1).to(device=device)


def pair_to_tensor(pair, input_dict, output_dict, device):
    input, output = pair
    input_tensor = sentence_to_tensor(input, input_dict, device)
    output_tensor = sentence_to_tensor(output, output_dict, device)
    return input_tensor, output_tensor


def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
