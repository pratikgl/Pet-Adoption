# Importing libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

ds = pd.read_csv("Dataset/train.csv")


ds= ds.drop('pet_id', axis=1)
ds=ds.drop('issue_date', axis=1)
ds=ds.drop('listing_date', axis=1)
ds=ds.drop('breed_category', axis=1)

ds=ds.drop('pet_category', axis=1)
ds=ds.drop('color_type', axis=1)

ds['condition']=ds['condition'].fillna(3)

# mean by target imputation
matrix = [0 for y in range(10)] 
count = [0 for y in range(10)] 
avg = [0 for y in range(10)]

for i in range(len(ds)):
    if ds['X1'][i] != 0:
        x = int(ds['X2'][i])
        matrix[x] = matrix[x] + ds['X1'][i]
        count[x] += 1
        
for i in range(10):
    if count[i] != 0:
        avg[i] = round(matrix[i]/count[i])

x3 = [0]*len(ds)
for i in range(len(ds)):
    if ds['X1'][i] == 0:
        x = int(ds['X2'][i])
        ds['X1'][i] = avg[x]

x3 = np.resize(x3, (len(ds), 1))

# hot encoding
integer_encoded = ds['condition'].values
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x1 = onehot_encoder.fit_transform(integer_encoded)
df = pd.DataFrame(x1,  columns=['C0','C1', 'C2','C3'])

ds = pd.concat([df, ds ], axis=1)

ds = ds.drop('condition', axis=1)

sns.heatmap(
    data = ds.corr(), #feeding data
    annot=True, #printing values on cells
    fmt='.2f', #rounding off
    cmap='RdYlGn' #colors
)
plt.title("Correlation Matrix before removing Zero")
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()