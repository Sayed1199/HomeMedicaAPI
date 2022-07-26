import pickle

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from pydantic import BaseModel


class BreastCancerData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    radius_se: float
    perimeter_se: float
    area_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


def predict(data: BreastCancerData):
    """
    method that uses the breastCancer.pkl model to predict breast Cancer
    :param data:BreastCancerData
    :return:int 0 | 1
    """
    with open('ModelsControllers/Models/breastCancer.pkl', 'rb') as pickled_model:
        model = pickle.load(pickled_model)
    values = np.array(list(data.dict().values())).reshape(1, -1)
    prediction = model.predict(values)[0]
    return prediction


def _formatResponse(prediction):
    """
    method to format the output of sklearn model (0 or 1) to a "PredictionResponse" object
    :param prediction:int (0|1)
    :return: PredictionResponse object
    """
    return PredictionResponse(Positive=prediction == 1)


def evaluateBreastCancer(data: BreastCancerData):
    """

    :param data: BreastCancerData
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction = predict(data)
    return _formatResponse(prediction)
