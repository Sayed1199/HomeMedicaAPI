import pickle

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from pydantic import BaseModel


class HeartDiseaseData(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_blood_pressure: int
    serum_cholestoral: int
    fasting_blood_sugar: int
    resting_electrocardiographic_results: int
    maximum_heart_rate_achieved: int
    exercise_induced_angina: int
    ST_depression_induced_by_exercise_relative_to_rest: int
    the_slope_of_the_peak_exercise_ST_segment: int
    number_of_major_vessels_colored_by_flourosopy: int
    target: int

    class Config:
        schema_extra = {
            "example": {
                "sex": "Male: 1, female: 0",
                "resting_blood_pressure": "in mm Hg",
                "serum_cholestoral": "in mg/dl",
                "fasting_blood_sugar": "120 mg/dl (1 = true; 0 = false)",
                "exercise_induced_angina": "(1 = yes; 0 = no)",
                "target": "3 = normal; 6 = fixed defect; 7 = reversable defect",

            }
        }


def predict(data: HeartDiseaseData):
    """
    method that uses the heart.pkl model to predict heart disease
    :param data:HeartDiseaseData
    :return:int 0 | 1
    """
    with open('ModelsControllers/Models/heart.pkl', 'rb') as pickled_model:
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


def evaluateHeartDisease(data: HeartDiseaseData):
    """

    :param data: HeartDiseaseData
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction = predict(data)
    return _formatResponse(prediction)
