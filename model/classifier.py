"""
Image classification logic built on transfer learning.

Uses MobileNetV2, pretrained on ImageNet (1,000 object categories), so no
training is required — we simply hand it a photo and ask for its opinion.
"""

import time

import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

TARGET_SIZE = (224, 224)  # fixed input size MobileNetV2 expects


def load_model():
    """Loads MobileNetV2 with ImageNet weights.

    The first call downloads ~14 MB of pretrained weights from the
    internet; after that they're cached locally by Keras.
    """
    return MobileNetV2(weights="imagenet")


def prepare_image(pil_image: Image.Image) -> np.ndarray:
    """Resizes a PIL image to 224x224 and converts it into the numeric
    batch format MobileNetV2 expects: (1, height, width, channels),
    scaled the same way the network was originally trained.
    """
    pil_image = pil_image.convert("RGB").resize(TARGET_SIZE)
    image_array = np.array(pil_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_batch)


def predict_image(model, pil_image: Image.Image, top: int = 5):
    """Runs a PIL image through the model and returns the top predictions.

    Returns:
        decoded: list of (label, probability) tuples, most likely first
        elapsed_ms: how long inference took, in milliseconds
    """
    processed = prepare_image(pil_image)

    start_time = time.time()
    predictions = model.predict(processed, verbose=0)
    elapsed_ms = (time.time() - start_time) * 1000

    decoded_predictions = decode_predictions(predictions, top=top)[0]
    decoded = [(label, float(prob)) for _, label, prob in decoded_predictions]

    return decoded, elapsed_ms
