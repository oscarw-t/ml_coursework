from sklearn.model_selection import train_test_split
import pandas
ds = pandas.read_csv('data/CW1_train.csv')

#ordinal mappings (worst to best) from GIA
"""
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}


ds['cut'] = ds['cut'].map(cut_map)
ds['color'] = ds['color'].map(color_map)
ds['clarity'] = ds['clarity'].map(clarity_map)
"""
#one hot encoding
ds = pandas.get_dummies(ds, columns=['cut', 'color', 'clarity'], drop_first=True)

#90/10 split
train_ds, test_ds = train_test_split(ds, test_size=0.1, random_state=42)

train_ds.to_csv('data/training_data_onehotencoded.csv', index=False)
test_ds.to_csv('data/testing_data_onehotencoded.csv', index=False)

print(f"training set: {train_ds.shape[0]} samples")
print(f"test set: {test_ds.shape[0]} samples")