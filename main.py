from ModelsControllers.responseController import PredictionResponse, HTTPError
from ModelsControllers.KidneyDiseaseController import KidneyDiseaseData, evaluateKidneyDisease
from ModelsControllers.breastcancerController import BreastCancerData, evaluateBreastCancer
from ModelsControllers.diabetesController import DiabetesData, evaluateDiabetes
from ModelsControllers.heartDiseaseController import HeartDiseaseData, evaluateHeartDisease
from ModelsControllers.liverDiseaseController import LiverDiseaseData, evaluateLiverDisease
from ModelsControllers.malariaDiseaseController import evaluateMalariaDisease
from ModelsControllers.pneumoniaDiseaseController import evaluatePneumoniaDisease
from ModelsControllers.covid19Controller import evaluateCovid19
from fastapi import FastAPI, Depends, UploadFile
from fastapi.exceptions import HTTPException

description = """
# HomeMedicaAPI 🚀

## Information about the Diseases the API support.


### [Diabetes](#/Diabetes)
Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy. Sometimes your body doesn’t make enough—or any—insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells.

**Symptoms**
* Urinating often.
* Feeling very thirsty.
* Extreme fatigue.
* Blurry vision.

<br>

### [Breast Cancer](#/BreastCancer)
Breast cancer is cancer that forms in the cells of the breasts. After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States. Breast cancer can occur in both men and women, but it's far more common in women.

**Symptoms**
* A breast lump or thickening that feels different from the surrounding tissue
* Change in the size, shape or appearance of a breast
* Changes to the skin over the breast, such as dimpling
* Redness or pitting of the skin over your breast, like the skin of an orange

<br>

### [Heart disease](#/HeartDisease)

<br>

### [Chronic kidney disease](#/KidneyDisease)
Chronic kidney disease, also called chronic kidney failure, describes the gradual loss of kidney function. Your kidneys filter wastes and excess fluids from your blood, which are then excreted in your urine. When chronic kidney disease reaches an advanced stage, dangerous levels of fluid, electrolytes and wastes can build up in your body.

**Symptoms**
* Nausea
* Vomiting
* Fatigue and weakness
* Muscle twitches and cramps

<br>
### [Liver disease](#/LiverDisease)
Symptoms of liver disease can vary, but they often include swelling of the abdomen and legs, bruising easily, changes in the color of your stool and urine, and jaundice, or yellowing of the skin and eyes. Sometimes there are no symptoms. Tests such as imaging tests and liver function tests can check for liver damage and help to diagnose liver diseases.
<br>

<br>

### [Malaria](#/MalariaDisease)
Malaria is a mosquito-borne infectious disease that affects humans and other animals. Malaria causes symptoms that typically include fever, tiredness, vomiting, and headaches. In severe cases it can cause yellow skin, seizures, coma, or death. Symptoms usually begin ten to fifteen days after being bitten by an infected mosquito. If not properly treated, people may have recurrences of the disease months later.

**Symptoms**
* Fever. This is the most common symptom.
* Chills
* Headache
* Nausea and vomiting
<br>

### [Pneumonia](#/PneumoniaDisease)
Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.

**Symptoms**
* *Cough*, which may produce greenish, yellow or even bloody mucus.
* *Fever*, sweating and shaking chills.
* *Shortness of breath*.
* *Rapid*, shallow breathing.

---

## Model Accuracies:
* Diabetes Model: 98.25%
* Breast Cancer Model: 98.25%
* Heart Disease Model: 85.25%
* Kidney Disease Model: 99%
* Liver Disease Model: 78%
* Malaria Model: 96%
* Pneumonia Model: 95%
"""

tags_metadata = [

    {
        "name": "Diabetes",
        "description": "Diabetes Predictor",
    },
    {
        "name": "BreastCancer",
        "description": "Breast Cancer Predictor",
    },
    {
        "name": "HeartDisease",
        "description": "Heart Disease Predictor",
    },
    {
        "name": "KidneyDisease",
        "description": "Kidney Disease Predictor",
    },
    {
        "name": "LiverDisease",
        "description": "Liver Disease Predictor",
    },

    {
        "name": "MalariaDisease",
        "description": "Malaria Predictor",
    },

    {
        "name": "PneumoniaDisease",
        "description": "Pneumonia Predictor",
    },
    {
        "name": "Covid19",
        "description": "Covid19 Predictor",
    },

]

app = FastAPI(
    title="HomeMedicaAPI",
    description=description,
    version="0.0.1",
    terms_of_service="",
    contact={
        "name": "HomeMedica Team",
        "url": "http://www.example.com/contact/",
        "email": "phenomenalboy0@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=tags_metadata
)


@app.get("/")
async def root():
    return ""


@app.get("/diabetes/",
         responses={
             200: {"model": PredictionResponse, "description": "Success", },
             400: {
                 "model": HTTPError,
                 "description": "Custom Model Error",
             },
         },
         tags=["Diabetes"])
async def predictDiabetes(query: DiabetesData = Depends()):
    try:
        return evaluateDiabetes(query)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.get("/breastCancer/",
         responses={
             200: {"model": PredictionResponse, "description": "Success", },
             400: {
                 "model": HTTPError,
                 "description": "Custom Model Error",
             },
         },
         tags=["BreastCancer"])
async def predictBreastCancer(query: BreastCancerData = Depends()):
    try:
        return evaluateBreastCancer(query)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.get("/heartDisease/",
         responses={
             200: {"model": PredictionResponse, "description": "Success", },
             400: {
                 "model": HTTPError,
                 "description": "Custom Model Error",
             },
         },
         tags=["HeartDisease"])
async def predictHeartDisease(query: HeartDiseaseData = Depends()):
    try:
        return evaluateHeartDisease(query)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.get("/KidneyDisease/",
         responses={
             200: {"model": PredictionResponse, "description": "Success", },
             400: {
                 "model": HTTPError,
                 "description": "Custom Model Error",
             },
         },
         tags=["KidneyDisease"])
async def predictKidneyDisease(query: KidneyDiseaseData = Depends()):
    try:
        return evaluateKidneyDisease(query)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.get("/liverDisease/",
         responses={
             200: {"model": PredictionResponse, "description": "Success", },
             400: {
                 "model": HTTPError,
                 "description": "Custom Model Error",
             },
         },
         tags=["LiverDisease"])
async def predictLiverDisease(query: LiverDiseaseData = Depends()):
    try:
        return evaluateLiverDisease(query)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.post("/malariaDisease/",
          responses={
              200: {"model": PredictionResponse, "description": "Success", },
              400: {
                  "model": HTTPError,
                  "description": "Custom Model Error",
              },
          },
          tags=["MalariaDisease"])
async def predictMalariaDisease(cell_image: UploadFile):
    try:
        request_object_content = await cell_image.read()
        return evaluateMalariaDisease(request_object_content)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.post("/pneumoniaDisease/",
          responses={
              200: {"model": PredictionResponse, "description": "Success", },
              400: {
                  "model": HTTPError,
                  "description": "Custom Model Error",
              },
          },
          tags=["PneumoniaDisease"])
async def predictPneumoniaDisease(xray_image: UploadFile):
    try:
        request_object_content = await xray_image.read()
        return evaluatePneumoniaDisease(request_object_content)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


@app.post("/covid19/",
          responses={
              200: {"model": PredictionResponse, "description": "Success", },
              400: {
                  "model": HTTPError,
                  "description": "Custom Model Error",
              },
          },
          tags=["Covid19"])
async def predictCovid19(xray_image: UploadFile):
    try:
        request_object_content = await xray_image.read()
        return evaluateCovid19(request_object_content)
    except Exception as e:
        print(e)
        raise HTTPException(400, detail=str(e))


#if __name__ == "__main__":
#   uvicorn.run("main:app", host="127.0.0.1", port=5500, log_level="info", reload=True, debug=True, workers=3)
