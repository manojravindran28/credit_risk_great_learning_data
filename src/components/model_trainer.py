import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class ModelTrainer:

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        candidates = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        }

        best_name, best_model, best_score = None, None, -1

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            score = f1_score(y_test, model.predict(X_test))
            print(f"{name} — F1: {score:.4f}")
            if score > best_score:
                best_name, best_model, best_score = name, model, score

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        print(f"\nBest model: {best_name} (F1: {best_score:.4f}) saved to artifacts/model.pkl")

