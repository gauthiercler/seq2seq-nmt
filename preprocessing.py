import re

from tokens import tokens


def normalize_string(s):
    s = s.lower()
    s = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ]+", r" ", s)
    s = s.strip()
    return s


class Dict:
    def __init__(self):
        self.dict = tokens
        self.max_id = len(self.dict)

    def add(self, word):
        if word not in self.dict:
            self.dict[word] = self.max_id
            self.max_id += 1

    def size(self):
        return len(self.dict)


def get_max_length(pairs):
    return max([len(s.split(' ')) for pair in pairs for s in pair])


def sentence_length_threshold(pair, max_length):
    print(pair)
    return all((len(p.split(' ')) <= max_length for p in pair))


def load_data(filename, max_length):
    content = open(filename, encoding='utf-8').read().strip().split('\n')
    pairs = []
    input_dict = Dict()
    output_dict = Dict()
    for line in content:
        pair = tuple([normalize_string(sentence) for sentence in line.split('\t')])
        if sentence_length_threshold(pair, max_length):
            [input_dict.add(w) for w in pair[0].split(' ')]
            [output_dict.add(w) for w in pair[1].split(' ')]
            pairs.append(pair)
    print(f'Loaded {len(pairs)} pairs')
    return pairs, input_dict, output_dict
