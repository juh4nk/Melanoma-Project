"""
preprocessing/image_preprocessing.py
======================================
Image transformation functions used inside the tf.data pipeline.

Called from data/dataset_loader.py via Dataset.map(). Functions are
grouped into two categories:

    Base preprocessing — applied to ALL splits (train, val, test):
        load_and_decode → resize → normalize

    Augmentation — applied to the TRAINING split only:
        random flips, rotation, brightness/contrast jitter

Note on ISIC artifacts:
    ISIC 2017 images may contain clinical markers (rulers, ink, hair).
    Augmentation partially mitigates their effect on generalization.
    Dedicated artifact removal is out of scope for v1.
"""

import tensorflow as tf


def load_and_decode(file_path: str) -> tf.Tensor:
    """
    Read a JPEG file from disk and decode it to a float32 RGB tensor.

    Uses tf.io.read_file + tf.image.decode_jpeg, forcing 3 channels.
    Output pixel values are in the range [0, 255].

    Args:
        file_path (str or tf.string): Absolute path to a .jpg image.

    Returns:
        tf.Tensor: Shape (H, W, 3), dtype float32.
    """
    pass


def resize(image: tf.Tensor, target_size: tuple) -> tf.Tensor:
    """
    Resize an image tensor to the target spatial dimensions.

    Uses tf.image.resize with bilinear interpolation.
    EfficientNetB0 expects (224, 224); adjust for larger variants.

    Args:
        image (tf.Tensor):    Input image, shape (H, W, 3).
        target_size (tuple):  Desired (height, width), e.g. (224, 224).

    Returns:
        tf.Tensor: Shape (target_h, target_w, 3).
    """
    pass


def normalize(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize pixel values to the range expected by the backbone.

    EfficientNetB0 includes its own normalization layer internally,
    so raw [0, 255] values are passed through unchanged for that backbone.
    If switching to MobileNet or ResNet, update this function to
    scale to [-1, 1] or [0, 1] as required by those models.

    Args:
        image (tf.Tensor): Float32 tensor, values in [0, 255].

    Returns:
        tf.Tensor: Normalized tensor, same shape as input.
    """
    pass


def preprocess(file_path: str, label: int, image_size: tuple) -> tuple:
    """
    Full base preprocessing pipeline for one (path, label) example.

    Passed to Dataset.map() for all three splits (train, val, test).
    Chains: load_and_decode → resize → normalize.

    Args:
        file_path (str):     Path to the image file.
        label (int):         Binary label (0 or 1), passed through unchanged.
        image_size (tuple):  Target (height, width) for resizing.

    Returns:
        tuple: (preprocessed image tensor, label)
    """
    pass


def augment(image: tf.Tensor, label: int) -> tuple:
    """
    Apply random augmentation transforms to a single training image.

    Each call may produce a different result (stochastic transforms).
    The label is passed through unchanged.

    Planned transforms:
        - random_flip_left_right
        - random_flip_up_down
        - random_brightness   (max_delta=0.2)
        - random_contrast     (lower=0.8, upper=1.2)
        - random rotation ±15° via tf.keras.layers.RandomRotation

    Args:
        image (tf.Tensor): Preprocessed image, shape (H, W, 3).
        label (int):       Binary label, unchanged by augmentation.

    Returns:
        tuple: (augmented image tensor, label)
    """
    pass