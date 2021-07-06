import argparse
import os

from model.train import train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr',
                        '--learningrate',
                        type=float,
                        help='Learning rate',
                        dest='learning_rate',
                        default=1e-3)
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        help='Batch size',
                        dest='batch_size',
                        default=64)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        help='Number of epochs',
                        dest='epochs',
                        default=10)
    # parser.add_argument('-d',
    #                     '--datapath',
    #                     type=str,
    #                     help='Absolute path to the data',
    #                     dest='data_path')
    return parser.parse_args()


if __name__ == '__main__':
    pass
