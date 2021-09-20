from typing import Tuple
import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def get_dataset(
        data_path: str, batch_size: int, input_shape: Tuple[int, int]
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    ds_train = image_dataset_from_directory(
        os.path.join(data_path, 'train'),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=True,
    )

    ds_val = image_dataset_from_directory(
        os.path.join(data_path, 'validation'),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=True,
    )

    ds_test = image_dataset_from_directory(
        os.path.join(data_path, 'test'),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=True,
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def augment(image, label):
        image = tf.image.random_brightness(image, max_delta=0.2)

        image = tf.image.random_contrast(image, lower=0.05, upper=0.35)

        image = tf.image.random_flip_left_right(image)

        return image, label

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(len(ds_train))
    ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_val = ds_val.prefetch(AUTOTUNE)

    ds_test = ds_test.prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test
