import cv2 as cv
import numpy as np

def load_image_and_convert_to_rgb(image_path: str) -> np.ndarray:
    """
    Load an image from the given file path and convert it to RGB format.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The RGB image as a 3D numpy array (H, W, 3).
    """

    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB )
    return image_rgb

def resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize an image to the specified width and height.

    Parameters:
        img (numpy.ndarray): The input image as a 3D numpy array (H, W, 3).
        width (int): The desired width of the output image.
        height (int): The desired height of the output image.

    Returns:
        numpy.ndarray: The resized image as a 3D numpy array (height, width, 3).
    """
    resized_img = cv.resize(img, (width, height))
    return resized_img

def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the formula:
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    Parameters:
        img (numpy.ndarray): The input image as a 3D numpy array (H, W, 3).

    Returns:
        numpy.ndarray: The grayscale image as a 2D numpy array (H, W).
    """
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    return gray

def single_channel_histogram_equalization(img: np.ndarray, channel: int) -> np.ndarray:
    """
    Given a RGB image "img", perform single channel (channel index "channel") histogram equalization to achieve image enhancement.

    Parameters:
        img (numpy.ndarray): The input image as a 3D numpy array (H, W, 3).
        channel (int): The index of the channel, e.g. 0, 1, 2

    Returns:
        numpy.ndarray: The new image of size (H, W, 3), where the "channel" index of it is the enhancement of "img" with single channel histogram equalization.
    """
    result = img.copy()
    ch = result[:, :, channel]

    hist, _ = np.histogram(ch.flatten(), bins=256, range=(0, 256))

    cdf = hist.cumsum()

    # Normalize 
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype(np.uint8)

    result[:, :, channel] = cdf[ch]

    return result

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Perform histogram equalization on each channel of a RGB image.

    Parameters:
        img (numpy.ndarray): The input RGB image as a 3D numpy array (H, W, 3).

    Returns:
        numpy.ndarray: The equalized RGB image as a 3D numpy array (H, W, 3).
    """
    result = img.copy()

    for channel in range(3):
        ch = result[:, :, channel]
        hist, _ = np.histogram(ch.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()

        # Normalize
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_norm = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_norm, 0).astype('uint8')

        result[:, :, channel] = cdf_final[ch]

    return result