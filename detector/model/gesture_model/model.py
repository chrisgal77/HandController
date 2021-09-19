import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub


def get_model(num_classes: int) -> keras.Sequential:
    base = tf.keras.Sequential(
        [
            hub.KerasLayer(
                "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5",
                trainable=True,
            )
        ]
    )

    model = keras.Sequential([
        base,
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


if __name__ == "__main__":
    model = get_model(num_classes=10)
    result = model(tf.ones((1, 224, 224, 3)))

    assert result.shape == (1, 10)
