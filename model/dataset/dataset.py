import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# TODO


class HandDataset(tfds.core.GeneratorBasedBuilder):
    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(256, 256, 3)),
                    "label": tfds.features.ClassLabel(names=["no", "yes"]),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):

        extracted_path = dl_manager.download_and_extract("http://data.org/data.zip")
        # dl_manager returns pathlib-like objects with `path.read_text()`,
        # `path.iterdir()`,...
        return {
            "train": self._generate_examples(path=extracted_path / "train_images"),
            "test": self._generate_examples(path=extracted_path / "test_images"),
        }

    def _generate_examples(self, path):
        for img_path in path.glob("*.jpeg"):
            yield img_path.name, {
                "image": img_path,
                "label": "yes" if img_path.name.startswith("yes_") else "no",
            }
