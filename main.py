import torch
from torch import nn

from argparser import parse_arguments
from decoder import Decoder, train_decoder
from encoder import Encoder, encode_seq
from preprocessing import load_data
import torch.optim as optim
from sklearn.model_selection import train_test_split

from tokens import tokens
from tqdm import tqdm

from utils import sentence_to_tensor, get_key, pair_to_tensor

from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_arguments()


def evaluate(encoder, decoder, sentence, source_lang, target_lang):
    out_seq = []
    with torch.no_grad():
        in_tensor = sentence_to_tensor(sentence, source_lang, device)

        ctx_vec = encode_seq(encoder, in_tensor, device=device)

        input = torch.tensor([[tokens['SOS']]]).to(device=device)
        hidden = ctx_vec

        for idx in range(args.max_words):
            output, hidden = decoder(input, hidden)
            topv, topi = output.topk(1)
            if topi.item() == tokens['EOS']:
                break
            else:
                out_seq.append(get_key(target_lang.dict, topi.item()))
            input = topi.squeeze().detach()

    return out_seq


def train_dev_test_split(pairs):
    train, test = train_test_split(pairs, test_size=args.sets_size[-1], shuffle=True)
    train, dev = train_test_split(train, test_size=sum(args.sets_size[1:]))

    return train, dev, test


def main():
    print(torch.cuda.is_available())
    print(args)

    pairs, source_lang, target_lang = load_data('data/eng-fra.txt', args.max_words)
    train, dev, test = train_dev_test_split(pairs)

    train_tensors = [pair_to_tensor(p, source_lang, target_lang, device) for p in train]
    dev_tensors = [pair_to_tensor(p, source_lang, target_lang, device) for p in dev]

    total_loss = 0

    encoder = Encoder(source_lang.size(), args.cv_size, args.max_words).to(device=device)
    decoder = Decoder(args.cv_size, target_lang.size(), args.use_attention, args.dropout,
                      args.teacher_forcing).to(device=device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate)

    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):

        tqdm_bar = tqdm(train_tensors)
        for idx, sample in enumerate(tqdm_bar):
            input, output = sample

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            ctx_vec = encode_seq(encoder, input, device=device)

            loss = train_decoder(decoder, ctx_vec, output, criterion, args.teacher_forcing, device)

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item() / output.size()[0]

            if idx % args.verbose_rate == 0:
                tqdm_bar.set_description(f'Epoch: {epoch + 1}/{args.epochs}, loss={(total_loss / args.verbose_rate):.3f}')
                total_loss = 0

    bleu = 0
    for pair in test:
        out_seq = evaluate(encoder, decoder, pair[0], source_lang, target_lang)
        bleu += sentence_bleu([pair[1].split(' ')], out_seq)
        print(f'source seq\t-> {pair[0]}')
        print(f'ground truth\t-> {pair[1]}')
        print(f'generated seq\t-> {" ".join(out_seq)}', end='\n\n')
    bleu /= len(test)
    print(f'Bleu score: {bleu}')

if __name__ == '__main__':
    main()
