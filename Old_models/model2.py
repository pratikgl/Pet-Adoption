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
        x3[i] = avg[x]
    else:
        x3[i] = ds['X1'][i]

x3 = np.resize(x3, (len(ds), 1))


# creating final dataframe
x_final = np.concatenate(
    (x1, 
     x2, 
     ds[['length(m)']], 
     ds[['height(cm)']], 
     x3, 
     ds[['X2']], 
     ds[['breed_category']], 
     ds[['pet_category']]), axis=1)

final_ds = pd.DataFrame(
    data=x_final[0:,0:], 
    index=[i for i in range(x_final.shape[0])],
    columns=['f'+str(i) for i in range(x_final.shape[1])])

'''
# correlation matrix
sns.heatmap(
    data = final_ds.corr(), #feeding data
    annot=True, #printing values on cells
    fmt='.2f', #rounding off
    cmap='RdYlGn' #colors
)
plt.title("Correlation Matrix before removing Zero")
fig = plt.gcf()
fig.set_size_inches(100, 80)
plt.show()
'''

# feature selction
feature_names = final_ds.columns[0:63]
x = final_ds[feature_names]
y = final_ds['f64']

x_new = SelectKBest(chi2, k=10).fit_transform(x, y)

X_train_1, X_validation_1, Y_train_1, Y_validation_1 = train_test_split(x_new, y, test_size=0.20, random_state=1) 

from sklearn.ensemble import GradientBoostingClassifier
clf1 = GradientBoostingClassifier(random_state=0)
clf1.fit(X_train_1, Y_train_1);
predictions1 = clf1.predict(X_validation_1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_validation_1, predictions1)

result = classification_report(Y_validation_1, predictions1)
print(result)

# ////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////

# working with the test data

# ////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////


test = pd.read_csv("Dataset/test.csv")

answer = pd.DataFrame(test['pet_id'])
answer['breed_category']= np.nan
answer['pet_category']= np.nan

# fill empty cells of test data
test['condition']=test['condition'].fillna(3)

# hot encoding
integer_encoded = test['condition'].values
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x1 = onehot_encoder.fit_transform(integer_encoded)

integer_encoded = test['color_type'].values
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x2 = onehot_encoder.fit_transform(integer_encoded)

# X1 imputation
x3 = list()
for i in range(len(test)):
    if(test['X1'][i] != 0):
        x3.append(test['X1'][i])
imputing_value = np.median(x3)
for i in range(len(test)):
    if(test['X1'][i] == 0):
        test['X1'][i] = imputing_value


# creating final dataframe
        

x_final = np.concatenate(
    (test[['pet_id']],
     x1, 
     x2, 
     test[['length(m)']], 
     test[['height(cm)']], 
     x3, 
     test[['X2']]), axis=1)

final_ds = pd.DataFrame(
    data = x_final[0:,0:], 
    index = [i for i in range(x_final.shape[0])],
    columns = ['f'+str(i) for i in range(x_final.shape[1])])











