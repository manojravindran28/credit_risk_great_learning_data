import pandas as pd
from src.components.data_transformations import DataTransformation
from src.components.model_trainer import ModelTrainer

df = pd.read_csv("data/Default.csv")

transform = DataTransformation()
df = transform.initiate_transformation(df)

X = df.drop("default", axis=1)
y = df["default"]

trainer = ModelTrainer()
trainer.initiate_model_training(X, y)

print("Training Completed")