import tensorflow as tf
from tensorflow import keras
import os

from .model import get_model
from ..dataset import get_dataset


def train(
    num_classes: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    data_path: str,
    logs_path: str,
) -> keras.Model:

    model = get_model(num_classes)

    ds_train, ds_val, ds_test = get_dataset(data_path, batch_size, (224, 224))

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logs_path, "logs/"),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logs_path, "checkpoints/checkpoint_best/"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
        ),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save_weights(os.path.join(logs_path, "checkpoint/last/"))

    return history
