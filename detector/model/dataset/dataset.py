from typing import Tuple

import tensorflow as tf


def get_dataset(data_path: str, batch_size: int, input_shape: Tuple[int, int]) -> Tuple[tf.data.Dataset,
                                                                                        tf.data.Dataset,
                                                                                        tf.data.Dataset]:
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=True,
    )

    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=True,
    )

    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=True,
    )

    return ds_train, ds_val, ds_test
