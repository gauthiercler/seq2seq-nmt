import re

from prepare import DataLoader
from tokens import Tokens


def normalize(s):
    for idx, w in enumerate(s):
        w = w.lower()
        w = re.sub(r"[!?.,']+", r" ", w)
        w = w.strip()
        s[idx] = w
    return s


class Dict:
    def __init__(self):
        self.dict = {k: Tokens[k].value for k in dir(Tokens) if not k.startswith('__')}
        self.max_id = len(self.dict)

    def add(self, word):
        if word not in self.dict:
            self.dict[word] = self.max_id
            self.max_id += 1

    def size(self):
        return len(self.dict)


def get_max_length(pairs):
    return max([len(s) for pair in pairs for s in pair])


def sentence_length_threshold(pair, max_length):
    return all((len(p) <= max_length for p in pair))


def load_data(lang, max_length):
    loader = DataLoader()
    loader.download_and_extract(lang)
    dataset = loader.prepare(lang)
    pairs = []
    input_dict = Dict()
    output_dict = Dict()
    for sample in dataset.examples:
        pair = (normalize(sample.source), normalize(sample.target))
        if sentence_length_threshold(pair, max_length):
            [input_dict.add(w) for w in pair[0]]
            [output_dict.add(w) for w in pair[1]]
            pairs.append(pair)
    print(f'Loaded {len(pairs)} pairs')
    return pairs, input_dict, output_dict
