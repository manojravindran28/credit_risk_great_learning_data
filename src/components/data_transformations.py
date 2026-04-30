import pandas as pd
import numpy as np
import pickle
import os

class DataTransformation:

    def initiate_transformation(self, df):

        # Outlier treatment
        Q1, Q3 = df['balance'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        UL = Q3 + 1.5 * IQR

        df['balance'] = np.where(df['balance'] > UL, UL, df['balance'])

        # Dummy Encoding
        df = pd.get_dummies(df, drop_first=True).astype(int)
        df.columns = ['balance', 'income', 'default', 'student']

        # Save upper limit
        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(UL, f)

        return df