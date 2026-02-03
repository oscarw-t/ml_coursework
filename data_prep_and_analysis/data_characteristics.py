import pandas

ds = pandas.read_csv('data/CW1_train.csv')

print (ds.shape)
#print(ds.head())
#print(ds.dtypes)
print("----------------------------------------------------")
print(ds.info())
print("----------------------------------------------------")
print(ds.describe())
print("----------------------------------------------------")
print(ds[['a1','a2','a3','a4','a5','b1','b2','b3','b4','b5']].describe())
print(ds[['a6','a7','a8','a9','a10','b6','b7','b8','b8','b10']].describe())
print("----------------------------------------------------")
print (ds[['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10',
        'b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','outcome']].corr()['outcome']) #.drop('outcome')
print("----------------------------------------------------")
print (ds[['carat','depth','table','price','x','y','z', 'outcome']].corr()['outcome'])