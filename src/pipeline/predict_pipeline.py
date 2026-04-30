import pickle
import pandas as pd
import numpy as np

class PredictPipeline:

    def __init__(self):

        with open("artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("artifacts/preprocessor.pkl", "rb") as f:
            self.UL = pickle.load(f)

    def predict(self, balance, income, student):

        # Apply preprocessing
        balance = min(balance, self.UL)

        df = pd.DataFrame({
            "balance": [balance],
            "income": [income],
            "student": [student]
        })

        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0][1]

        return pred, prob