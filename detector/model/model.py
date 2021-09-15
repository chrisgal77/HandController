import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub


def get_model(num_classes: int) -> keras.Sequnetial:
    base = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5", trainable=True)
    ])

    model = keras.Sequential([
        base,
        layers.Dense(32),
        layers.Dense(num_classes)
    ])

    return model