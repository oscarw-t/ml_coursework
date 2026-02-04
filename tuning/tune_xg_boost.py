import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


def xg_boost(hyperparams, X_train, y_train, X_val, y_val):
    model = XGBRegressor(**hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)
    return r2, train_r2



if __name__ == "__main__":


    ds = pd.read_csv('data/training_data_onehotencoded.csv')

    #log transform price CHANGE
    if 'price' in ds.columns:
        ds['price'] = np.log1p(ds['price'])



    X = ds.drop(columns=['outcome'])
    y = ds['outcome']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    current_best_hyperparameters = [
        {
            "n_estimators": 2000, 
            "learning_rate": 0.01, 
            "max_depth": 2, 
            "subsample": 0.4, 
            "colsample_bytree": 0.4, 
            "min_child_weight": 150,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            #"random_state": 42
        }
        ]





    bestr2 = 0
    besthypers = []


    #use hyperparameters or current_best_hyperparameters
    for hyperparams in current_best_hyperparameters:
        r2, train_r2 = xg_boost(hyperparams, X_train, y_train, X_val, y_val)
        print("hyperparam set tested")
        if r2>bestr2:
            bestr2=r2
            besthypers=hyperparams

        print(hyperparams)
        print(f"train r2 {train_r2}")
        print(f"val r2 {r2}")

    print(bestr2)
    print(besthypers)


##current best hyperparams {'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 2, 'subsample': 0.4, 'colsample_bytree': 0.4, 'min_child_weight': 150, 'reg_alpha': 2.0, 'reg_lambda': 10.0, 'random_state': 42}