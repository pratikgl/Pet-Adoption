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

# Load data
dataset = pd.read_csv("Dataset/train.csv")
ds = dataset

# converting Nan values
ds['condition']=ds['condition'].fillna(3)

# hot encoding
integer_encoded = ds['condition'].values
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x1 = onehot_encoder.fit_transform(integer_encoded)

integer_encoded = ds['color_type'].values  
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x2 = onehot_encoder.fit_transform(integer_encoded)

# mean by target imputation
matrix = [[0 for x in range(5)] for y in range(5)] 
count = [[0 for x in range(5)] for y in range(5)] 
avg = [[0 for x in range(5)] for y in range(5)] 

for i in range(len(ds)):
    if ds['X1'][i] != 0:
        x = int(ds['breed_category'][i])
        y = int(ds['pet_category'][i])
        matrix[x][y] = matrix[x][y] + ds['X1'][i]
        count[x][y] += 1
        
for i in range(5):
    for j in range(5):
        if count[i][j] != 0:
            avg[i][j] = round(matrix[i][j]/count[i][j])

x3 = [0]*len(ds)
for i in range(len(ds)):
    if ds['X1'][i] == 0:
        x = int(ds['breed_category'][i])
        y = int(ds['pet_category'][i])
        x3[i] = avg[x][y]
    else:
        x3[i] = ds['X1'][i]

x3 = np.resize(x3, (len(ds), 1))


# creating final dataframe
x_final = np.concatenate(
    (x1, 
     x2, 
     ds[['length(m)']], 
     ds[['height(cm)']], 
     x3, ds[['X2']], 
     ds[['breed_category']], 
     ds[['pet_category']]), axis=1)

final_ds = pd.DataFrame(
    data=x_final[0:,0:], 
    index=[i for i in range(x_final.shape[0])],
    columns=['f'+str(i) for i in range(x_final.shape[1])])


# correlation matrix
sns.heatmap(
    data = final_ds.corr(), #feeding data
    annot=True, #printing values on cells
    fmt='.2f', #rounding off
    cmap='RdYlGn' #colors
)
plt.title("Correlation Matrix before removing Zero")
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()











