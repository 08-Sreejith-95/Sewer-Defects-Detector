# preprocess.py
import numpy as np
from PIL import Image
import io



import io
import numpy as np
from PIL import Image


def preprocess_pil(
    image: Image.Image,
    size: int,
    mean,
    std,
) -> np.ndarray:

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    image = image.convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)

    img_array = np.array(image, dtype=np.float32) / 255.0

    img_array = (img_array - mean) / std

    img_array = img_array.transpose(2, 0, 1)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_path(
    image_path: str,
    size: int,
    mean,
    std,
) -> np.ndarray:

    image = Image.open(image_path)

    return preprocess_pil(image, size, mean, std)


def preprocess_image_bytes(
    image_bytes: bytes,
    size: int,
    mean,
    std,
) -> np.ndarray:

    image = Image.open(io.BytesIO(image_bytes))

    return preprocess_pil(image, size, mean, std)