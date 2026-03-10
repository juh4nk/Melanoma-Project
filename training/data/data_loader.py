"""
data/dataset_loader.py
======================
Loads the ISIC 2017 dataset from disk and produces tf.data.Dataset
objects for the train, validation, and test splits.

Expected dataset layout on disk:
    data/
    ├── train/
    │   ├── images/     (.jpg files)
    │   └── labels.csv  (columns: image_id, melanoma)
    ├── val/
    │   ├── images/
    │   └── labels.csv
    └── test/
        ├── images/
        └── labels.csv

Binary label convention:
    1 → melanoma
    0 → non-melanoma
"""

import tensorflow as tf


def load_labels(csv_path: str) -> dict:
    """
    Parse an ISIC ground-truth CSV into a {image_id: label} dictionary.

    Reads the CSV, extracts the image_id and melanoma columns, and
    returns a flat mapping used to pair each image with its binary label.

    Args:
        csv_path (str): Path to the ISIC ground-truth CSV file.

    Returns:
        dict: { image_id (str): binary_label (int) }
              1 = melanoma, 0 = non-melanoma.
    """
    pass


def build_file_list(images_dir: str, labels: dict) -> tuple:
    """
    Pair each image file on disk with its binary label.

    Scans images_dir for .jpg files, looks each image_id up in the
    labels dict, and returns two parallel lists of equal length.

    Args:
        images_dir (str): Directory containing .jpg image files.
        labels (dict):    Output of load_labels().

    Returns:
        tuple: (file_paths: list[str], labels: list[int])
    """
    pass


def make_tf_dataset(
    file_paths: list,
    labels: list,
    image_size: tuple,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset pipeline from file paths and labels.

    Pipeline order:
        1. Dataset from (path, label) pairs
        2. Shuffle                    — training split only
        3. Map: preprocess()          — load, resize, normalize (via preprocessing module)
        4. Map: augment()             — random transforms, training split only
        5. Batch → Prefetch

    Args:
        file_paths (list):  Image file path strings.
        labels (list):      Parallel binary label list (0 or 1).
        image_size (tuple): Target (height, width), e.g. (224, 224).
        batch_size (int):   Number of samples per batch.
        augment (bool):     Apply augmentation — True for training split only.
        shuffle (bool):     Shuffle each epoch — True for training split only.

    Returns:
        tf.data.Dataset: Batched, prefetched dataset ready for model.fit().
    """
    pass


def get_datasets(config: dict) -> tuple:
    """
    Top-level function — returns all three dataset splits.

    Calls load_labels(), build_file_list(), and make_tf_dataset()
    for train, val, and test using paths and settings from config.

    Args:
        config (dict): Must contain dataset paths, image_size,
                       batch_size, and augment flag.

    Returns:
        tuple: (train_ds, val_ds, test_ds) as tf.data.Dataset objects.
    """
    pass