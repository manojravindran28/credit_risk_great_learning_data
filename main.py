import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformations import DataTransformation
from src.components.model_trainer import ModelTrainer

df = pd.read_csv("data/Default.csv")

transform = DataTransformation()
df = transform.initiate_transformation(df)

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21, stratify=y
)

trainer = ModelTrainer()
trainer.initiate_model_training(X_train, X_test, y_train, y_test)

print("Training Completed")