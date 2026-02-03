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
    return r2



if __name__ == "__main__":
    ds = pd.read_csv('data/training_data_encoded.csv')

    #log transform price CHANGE
    if 'price' in ds.columns:
        ds['price'] = np.log1p(ds['price'])



    X = ds.drop(columns=['outcome'])
    y = ds['outcome']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    hyperparameters = [
        {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "random_state": 42
        },
        {
        "n_estimators": 2000, 
        "learning_rate": 0.03, 
        "max_depth": 6, 
        "subsample": 0.9, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "random_state": 42
        },
        {
        "n_estimators": 4000, 
        "learning_rate": 0.01, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "random_state": 42
        },
        
        {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 8, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 3,
        "random_state": 42
        },
        {
        "n_estimators": 1500, 
        "learning_rate": 0.03, 
        "max_depth": 10, 
        "subsample": 0.7, 
        "colsample_bytree": 0.7, 
        "min_child_weight": 1,
        "random_state": 42
        },
        {
        "n_estimators": 800, 
        "learning_rate": 0.1, 
        "max_depth": 4, 
        "subsample": 0.9, 
        "colsample_bytree": 0.9, 
        "min_child_weight": 10,
        "random_state": 42
        },
        {
        "n_estimators": 500, 
        "learning_rate": 0.1, 
        "max_depth": 5, 
        "subsample": 0.75, 
        "colsample_bytree": 0.75, 
        "min_child_weight": 5,
        "random_state": 42
        },
        {
        "n_estimators": 300, 
        "learning_rate": 0.2, 
        "max_depth": 4, 
        "subsample": 0.85, 
        "colsample_bytree": 0.85, 
        "min_child_weight": 8,
        "random_state": 42
        },
        
        {
        "n_estimators": 5000, 
        "learning_rate": 0.005, 
        "max_depth": 7, 
        "subsample": 0.6, 
        "colsample_bytree": 0.6, 
        "min_child_weight": 5,
        "random_state": 42
    },
    {
        "n_estimators": 3000, 
        "learning_rate": 0.02, 
        "max_depth": 5, 
        "subsample": 0.95, 
        "colsample_bytree": 0.95, 
        "min_child_weight": 15,
        "random_state": 42
        },
        
        {
        "n_estimators": 1500, 
        "learning_rate": 0.05, 
        "max_depth": 7, 
        "subsample": 0.85, 
        "colsample_bytree": 0.85, 
        "min_child_weight": 7,
        "random_state": 42
        },
        {
        "n_estimators": 1200, 
        "learning_rate": 0.04, 
        "max_depth": 6, 
        "subsample": 0.75, 
        "colsample_bytree": 0.75, 
        "min_child_weight": 6,
        "random_state": 42
        },
        
        {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "reg_alpha": 0.1, 
        "reg_lambda": 1.0, 
        "random_state": 42
        },
        {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "random_state": 42
    },
    {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42
        },  
    
    
        {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "gamma": 0.1,
        "random_state": 42
        },
        {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 5,
        "gamma": 0.5,
        "random_state": 42
        },

        {
        "n_estimators": 2000, 
        "learning_rate": 0.03, 
        "max_depth": 8, 
        "subsample": 0.6, 
        "colsample_bytree": 0.6, 
        "min_child_weight": 3,
        "random_state": 42
        },
        
        {
        "n_estimators": 800, 
        "learning_rate": 0.02, 
        "max_depth": 5, 
        "subsample": 0.95, 
        "colsample_bytree": 0.95, 
        "min_child_weight": 10,
        "random_state": 42
        },
        
        {
        "n_estimators": 200, 
        "learning_rate": 0.3, 
        "max_depth": 3, 
        "subsample": 0.5, 
        "colsample_bytree": 0.5, 
        "min_child_weight": 1,
        "random_state": 42
        },
        
        {
        "n_estimators": 3000, 
        "learning_rate": 0.01, 
        "max_depth": 4, 
        "subsample": 0.7, 
        "colsample_bytree": 0.7, 
        "min_child_weight": 20,
        "random_state": 42
        },
        
        {
        "n_estimators": 2500, 
        "learning_rate": 0.025, 
        "max_depth": 7, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "min_child_weight": 8,
        "reg_alpha": 0.3,
        "reg_lambda": 0.7,
        "gamma": 0.2,
        "random_state": 42
        }
    ]


    bestr2 = 0
    besthypers = []
    for hyperparams in hyperparameters:
        r2 = xg_boost(hyperparams, X_train, y_train, X_val, y_val)
        if r2>bestr2:
            bestr2=r2
            besthypers=hyperparams

    print(bestr2)
    print(besthypers)
