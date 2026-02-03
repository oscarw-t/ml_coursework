import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('data/training_data_encoded.csv')

#split features and target
X = df.drop(columns=['outcome'])
y = df['outcome']

#train/val 90/10 split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

#
y_pred = model.predict(X_val)

#
r2 = r2_score(y_val, y_pred)
print(f"R² Score: {r2:.4f}")