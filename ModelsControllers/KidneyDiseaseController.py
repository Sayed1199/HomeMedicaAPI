import pickle

import numpy as np
from ModelsControllers.responseController import PredictionResponse
from pydantic import BaseModel


class KidneyDiseaseData(BaseModel):
    """
        Attribute Information:

        We use 24 + class = 25 ( 11 numeric ,14 nominal)
        1.Age(numerical)
        age in years
        2.Blood Pressure(numerical)
        bp in mm/Hg
        3.Specific Gravity(nominal)
        sg - (1.005,1.010,1.015,1.020,1.025)
        4.Albumin(nominal)
        al - (0,1,2,3,4,5)
        5.Sugar(nominal)
        su - (0,1,2,3,4,5)
        6.Red Blood Cells(nominal)
        rbc - (normal,abnormal)
        7.Pus Cell (nominal)
        pc - (normal,abnormal)
        8.Pus Cell clumps(nominal)
        pcc - (present,notpresent)
        9.Bacteria(nominal)
        ba - (present,notpresent)
        10.Blood Glucose Random(numerical)
        bgr in mgs/dl
        11.Blood Urea(numerical)
        bu in mgs/dl
        12.Serum Creatinine(numerical)
        sc in mgs/dl
        13.Sodium(numerical)
        sod in mEq/L
        14.Potassium(numerical)
        pot in mEq/L
        15.Hemoglobin(numerical)
        hemo in gms
        16.Packed Cell Volume(numerical)
        17.White Blood Cell Count(numerical)
        wc in cells/cumm
        18.Red Blood Cell Count(numerical)
        rc in millions/cmm
        19.Hypertension(nominal)
        htn - (yes,no)
        20.Diabetes Mellitus(nominal)
        dm - (yes,no)
        21.Coronary Artery Disease(nominal)
        cad - (yes,no)
        22.Appetite(nominal)
        appet - (good,poor)
        23.Pedal Edema(nominal)
        pe - (yes,no)
        24.Anemia(nominal)
        ane - (yes,no)
        25.Class (nominal)
        class - (ckd,notckd)

        """
    age: int
    bp: float
    al: float
    su: int
    rbc: int
    pc: int
    pcc: int
    ba: int
    bgr: float
    bu: float
    sc: float
    pot: float
    wc: float
    htn: int
    dm: int
    cad: int
    pe: int
    ane: int

    class Config:
        schema_extra = {
            "example": {
                "age": "age in years",
                "bp": "Blood Pressure(numerical) in mm/Hg",
                "al": "Albumin(nominal) - (0,1,2,3,4,5)",
                "su": "Sugar(nominal) - (0,1,2,3,4,5)",
                "rbc": "Red Blood Cells(nominal) - (normal,abnormal)",
                "pc": "Pus Cell (nominal) - (normal,abnormal)",
                "pcc": "Pus Cell clumps(nominal) - (present,notpresent)",
                "ba": "Bacteria(nominal) - (present,notpresent)",
                "bgr": "Blood Glucose Random(numerical) - in mgs/dl",
                "bu": "Blood Urea(numerical) - in mgs/dl",
                "sc": "Serum Creatinine(numerical) - in mgs/dl",
                "pot": "Potassium(numerical) - in mEq/L",
                "wc": "White Blood Cell Count(numerical) - in cells/cumm",
                "htn": "Hypertension(nominal) - (yes,no)",
                "dm": "Diabetes Mellitus(nominal)- (yes,no)",
                "cad": "Coronary Artery Disease(nominal) - (yes,no)",
                "pe": "Pedal Edema(nominal) - (yes,no)",
                "ane": "Anemia(nominal) - (yes,no)",

            }
        }


def predict(data: KidneyDiseaseData):
    """
    method that uses the heart.pkl model to predict Kidney disease
    :param data:KidneyDiseaseData
    :return:int 0 | 1
    """
    with open('ModelsControllers/Models/kidney.pkl', 'rb') as pickled_model:
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


def evaluateKidneyDisease(data: KidneyDiseaseData):
    """

    :param data: KidneyDiseaseData
    :return: PredictionResponse (positive : Ture or False)
    """

    prediction = predict(data)
    return _formatResponse(prediction)
