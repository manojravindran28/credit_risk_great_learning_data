import pickle
import os
from sklearn.linear_model import LogisticRegression

class ModelTrainer:

    def initiate_model_training(self, X, y):

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)

        print("Model Saved")

