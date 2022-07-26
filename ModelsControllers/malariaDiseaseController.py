import io

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from PIL import Image
from keras.models import load_model


def _adjust_image(image_file):
    """
    :param image_file: test image in a file-like object
    :return: image as np float64
    """
    img = Image.open(io.BytesIO(image_file))
    img = img.resize((36, 36))
    img = np.asarray(img)
    img = img.reshape((1, 36, 36, 3))
    img = img.astype(np.float64)
    return img


def predict(image_file):
    """
    method that uses the malaria.h5 model to predict malaria disease
    :param image_file: test image in a file-like object
    :return:int 0 | 1
    """
    img = _adjust_image(image_file)
    model = load_model("ModelsControllers/Models/malaria.h5")
    pred = np.argmax(model.predict(img)[0])
    return pred


def _formatResponse(prediction):
    """
    method to format model output (0 or 1) to a "PredictionResponse" object
    :param prediction:int (0|1)
    :return: PredictionResponse object
    """
    return PredictionResponse(Positive=prediction == 1)


def evaluateMalariaDisease(image_file):
    """

    :param image_file: test image in a file-like object
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction = predict(image_file)
    return _formatResponse(prediction)
