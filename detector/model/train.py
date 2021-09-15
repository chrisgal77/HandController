import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import *
from model import get_model


def train(num_classes: int,
          learning_rate: float,
          epochs: int,
          batch_size: int) -> keras.Model:

    model = get_model(num_classes)

    model.compile(optimizer= keras.optimizers.Adam(learning_rate=learning_rate),
                  loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  )


if __name__ == "__main__":
    train(
        NUM_CLASSES,
        LEARNING_RATE,
        EPOCHS,
        BATCH_SIZE
    )