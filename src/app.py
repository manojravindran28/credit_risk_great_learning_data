from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

pipeline = PredictPipeline()

class InputData(BaseModel):
    balance: float
    income: float
    student: int

@app.get("/")
def home():
    return {"message": "Credit Risk API Running"}

@app.post("/predict")
def predict(data: InputData):

    pred, prob = pipeline.predict(
        data.balance,
        data.income,
        data.student
    )

    return {
        "prediction": int(pred),
        "probability": round(float(prob),4)
    }

