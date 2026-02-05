import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np




def predict(inputs):
    model = XGBRegressor()
    model.load_model('saved_models/xg_boost_final.ubj')
    return model.predict(inputs)


def predict_test_outcome():
    
    test_data = pd.read_csv('data/CW1_test.csv')
    
    if 'outcome' in test_data.columns:
        test_data = test_data.drop(columns=['outcome'])

    test_data = pd.get_dummies(test_data, columns=['cut', 'color', 'clarity'], drop_first=True)

    predictions = predict(test_data)
    
    pd.DataFrame({'outcome': predictions}).to_csv('predictions_1.csv', index=False)


if __name__ == "__main__":
    predict_test_outcome()
    print("done")
