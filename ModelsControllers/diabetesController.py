import pickle

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from pydantic import BaseModel


class DiabetesData(BaseModel):
    Number_of_Pregnancies: int
    Glucose: float
    Blood_Pressure: float
    Skin_Thickness: float
    Insulin_Level: float
    Body_Mass_Index: float
    Diabetes_Pedigree_Function: float
    Age: int


def predict(data: DiabetesData):
    """
    method that uses the diabetes.pkl model to predict diabetes
    :param data:
    :return:int 0 | 1
    """
    with open('ModelsControllers/Models/diabetes.pkl', 'rb') as pickled_model:
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
    return PredictionResponse(Positive = prediction == 1)


def evaluateDiabetes(data: DiabetesData):
    """

    :param data: DiabetesData (Number_of_Pregnancies: int
    , Glucose: float
    , Blood_Pressure: float
    , Skin_Thickness: float
    , Insulin_Level: float
    , Body_Mass_Index: float
    , Diabetes_Pedigree_Function: float
    , Age: int)
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction = predict(data)
    return _formatResponse(prediction)
