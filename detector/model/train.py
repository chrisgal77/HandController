import argparse
import os
import logging

from gesture_model import train


def get_args():
    parser = argparse.ArgumentParser(
        description="Model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        help="Learning rate",
        dest="lr",
        default=0.001
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Epochs",
        dest="epochs",
        default=50
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="Batch size",
        dest="batch_size",
        default=32
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to data",
        dest="data_path"
    )

    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        help="Number of classes",
        dest="num_classes",
        default=10,
    )

    parser.add_argument(
        '-lo',
        '--logs',
        type=str,
        help='Path for logs and checkpoints',
        dest='logs_path',
        default=str(os.path.dirname(__file__))
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    train(
        num_classes=args.num_classes,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        logs_path=args.logs_path
    )
