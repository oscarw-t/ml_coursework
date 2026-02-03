import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



xg_boost_hyperparamters =        {
            "n_estimators": 2000, 
            "learning_rate": 0.01, 
            "max_depth": 2, 
            "subsample": 0.4, 
            "colsample_bytree": 0.4, 
            "min_child_weight": 150,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "random_state": 42
        }

random_forest_hyperparameters = {'n_estimators': 200,
                                 'random_state': 42,
                                 'max_depth': 10
                                }



def xg_boost(hyperparams, X_train, y_train, X_val):
    model = XGBRegressor(**hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    return y_pred



def random_forest(hyperparams, X_train, y_train, X_val):
    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    return y_pred



if __name__ == "__main__":
    
    ds = pd.read_csv('data/training_data_onehotencoded.csv')
            
    X = ds.drop(columns=['outcome'])
    y = ds['outcome']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


    count = 0
    while count < 1:

        rf_y = random_forest(random_forest_hyperparameters, X_train, y_train, X_val)
        xgb_y = xg_boost(xg_boost_hyperparamters, X_train, y_train, X_val)

        #y_hat = (rf_y + xgb_y) / 2
        y_hat = (0.1 * rf_y + 0.9 * xgb_y)

        print(r2_score(y_val, xgb_y))
        print(r2_score(y_val, rf_y))
        print(f"together :   {r2_score(y_val, y_hat)}")

        count+=1 