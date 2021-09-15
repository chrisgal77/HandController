from tensorflow import keras

from .model import get_model
from dataset import get_dataset


def train(
    num_classes: int, learning_rate: float, epochs: int, batch_size: int, data_path: str
) -> keras.Model:

    model = get_model(num_classes)

    ds_train, ds_val, ds_test = get_dataset(data_path, batch_size, (224, 224))

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
