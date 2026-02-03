import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


ds = pd.read_csv('data/training_data_encoded.csv')

X = ds.drop(columns=['outcome'])
y = ds['outcome']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)
print(f"XGBoost R² Score: {r2:.4f}")
