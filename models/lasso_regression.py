import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('data/training_data_encoded.csv')

X = df.drop(columns=['outcome'])
y = df['outcome']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


#change alpha
alpha = 0.0001
bestr2 = 0
bestalpha = 0
while alpha<1:
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    #score
    r2 = r2_score(y_val, y_pred)
    if r2>bestr2:
        bestr2 = r2
        bestalpha = alpha
    alpha+=0.001

print(bestr2)
print(bestalpha)

#current best alpha 0.0061, R2 score of 0.30264



"""
#check feature selection
feature_names = X.columns
coefficients = model.coef_
print("features kept by lasso")
print("-" * 40)
for name, coef in zip(feature_names, coefficients):
    if coef != 0:
        print(f"{name}: {coef:.4f}")

# 
n_zeros = sum(coefficients == 0)
print(f"dropped features {n_zeros}/{len(coefficients)}")

#predict
y_pred = model.predict(X_val)
#score
r2 = r2_score(y_val, y_pred)
print(f"R² Score: {r2:.4f}")

"""