from tensorflow import keras

from HandController.detector.model.model.config import *
from model import get_model
from dataset import get_data


def train(
    num_classes: int, learning_rate: float, epochs: int, batch_size: int, data_path: str
) -> keras.Model:

    model = get_model(num_classes)

    ds_train, ds_val, ds_test = get_data(data_path)

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
    )

    return history
