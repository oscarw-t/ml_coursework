import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

hyperparameters = [
    {'n_estimators': 100},
    {'n_estimators': 200},
    {'n_estimators': 500},
    {'n_estimators': 200, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 20},
    {'n_estimators': 200, 'max_depth': 30},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 4},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 8}

]








ds = pd.read_csv('data/training_data_encoded.csv')

X = ds.drop(columns=['outcome'])
y = ds['outcome']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)

print(f"R² Score: {r2:.4f}")



for hyperparameter_set in hyperparameters:
    model = RandomForestRegressor(random_state=42, **hyperparameter_set)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train) 
    train_r2 = r2_score(y_train, y_train_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{hyperparameter_set}: val R² = {r2:.4f}, train R2 = {train_r2:.4f}")
