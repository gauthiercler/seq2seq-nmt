import argparse


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.HelpFormatter):
    pass


def parse_arguments():
    argparser = argparse.ArgumentParser(formatter_class=lambda prog: CustomFormatter(prog, max_help_position=5))
    argparser.add_argument('--learning-rate', type=float, default=0.01, help='step size toward minimum of loss')
    argparser.add_argument('--epochs', type=int, default=10, help='number of epochs to train on dataset')
    argparser.add_argument('--dropout', type=float, default=0.1, help='probability to apply dropout for regularization')
    argparser.add_argument('--max-words', type=int, default=3, help='maximum number of words by sentence')
    argparser.add_argument('--cv-size', type=int, default=256, help='size of the context vector to represent source '
                                                                    'sequence')
    argparser.add_argument('--use-attention', action='store_true', help='use attention mechanism in decoder')
    argparser.add_argument('--verbose-rate', default=10, type=int, help='print interval')
    argparser.add_argument('--sets-size', default=[.8, .1, .1], nargs='+', type=float, help='percentage for train, '
                                                                                            'dev and test sets')
    argparser.add_argument('--teacher-forcing', choices=['beam-search', 'curriculum'],
                           default='curriculum', help='teacher forcing technique to use')
    args = argparser.parse_args()

    if sum(args.sets_size) > 1:
        argparser.print_usage()
        print("Sum of set percentages %s must be lower or equal to 1" % args.sets_size)
        argparser.exit()

    return args
