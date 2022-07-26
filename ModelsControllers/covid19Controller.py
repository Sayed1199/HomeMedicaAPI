import io

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from PIL import Image
from keras.models import load_model
import cv2

def _adjust_image(image_file):
    """
    :param image_file: test image in a file-like object
    :return: Image
    """
    img_size = 100
    img = Image.open(io.BytesIO(image_file))
    img = np.array(img)
    if (img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray / 255
    resized = cv2.resize(gray, (img_size, img_size))
    reshaped = resized.reshape(1, img_size, img_size)

    return reshaped


def predict(image_file):
    """
    method that uses the pneumonia.h5 model to predict pneumonia disease
    :param image_file: test image in a file-like object
    :return: int 0 | 1
    """
    img = _adjust_image(image_file)
    model = load_model("ModelsControllers/Models/model-015.model")
    prediction = model.predict(img)

    result = np.argmax(prediction, axis=1)[0]

    accuracy = float(np.max(prediction, axis=1)[0])
    return result, accuracy


def _formatResponse(prediction, accuracy):
    """
    method to format model output (0 or 1) to a "PredictionResponse" object
    :param prediction:int (0|1)
    :param accuracy:float
    :return: PredictionResponse object
    """
    return PredictionResponse(Positive=prediction == 1, Accuracy=accuracy)


def evaluateCovid19(image_file):
    """

    :param image_file: test image in a file-like object
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction, accuracy = predict(image_file)
    return _formatResponse(prediction, accuracy)
