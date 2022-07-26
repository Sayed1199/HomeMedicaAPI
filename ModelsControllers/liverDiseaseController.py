import pickle

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from pydantic import BaseModel


class LiverDiseaseData(BaseModel):
    Age: int
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphotase: float
    Alamine_Aminotransferase: float
    Aspartate_Aminotransferase: float
    Total_Protiens: float
    Albumin: int
    Albumin_and_Globulin_Ratio: float
    Gender: str

    class Config:
        schema_extra = {
            "example": {
                "age": "age in years",

            }
        }


def predict(data: LiverDiseaseData):
    """
    method that uses the liver.pkl model to predict Liver disease
    :param data:LiverDiseaseData
    :return:int 0 | 1
    """
    with open('ModelsControllers/Models/liver.pkl', 'rb') as pickled_model:
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


def evaluateLiverDisease(data: LiverDiseaseData):
    """

    :param data: LiverDiseaseData
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction = predict(data)
    return _formatResponse(prediction)
