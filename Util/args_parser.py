import argparse

__author__ = 'Manoel Ribeiro'


def ld_model_parser():
    parser = argparse.ArgumentParser(description='LD-models')
    parser.add_argument('-d', '--dataTrain', help='Data train', required=True)
    parser.add_argument('-l', '--seqTrain', help='Seq Train', required=True)
    parser.add_argument('-D', '--dataTest', help='Data Test', required=True)
    parser.add_argument('-L', '--seqTest', help='Seq Test', required=True)
    parser.add_argument('-m', '--model', help='Model', required=True)
    parser.add_argument('-s', '--stats', help='Stats', required=True)
    parser.add_argument('-r', '--result', help='Result', required=True)

    return vars(parser.parse_args())
