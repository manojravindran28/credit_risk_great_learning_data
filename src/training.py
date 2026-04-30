import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load data
Default = pd.read_csv("data/Default.csv")


Q1, Q3 = Default['balance'].quantile([.25, .75])
IQR = Q3 - Q1
LL = Q1 - 1.5*(IQR)
UL = Q3 + 1.5*(IQR)
df = Default[Default['balance'] > UL]
Default['balance'] = np.where(Default['balance'] > UL, UL, Default['balance'] )

Default = pd.get_dummies(Default, drop_first = True).astype(int)
Default.columns = ['balance', 'income', 'default', 'student']

X = Default.drop("default", axis=1)
y = Default["default"]

# Train model
model = LogisticRegression(max_iter=100)
model.fit(X, y)

# Save model
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved")


