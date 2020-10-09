# Importing libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer

# Load data
dataset = pd.read_csv("Dataset/train.csv")
ds = dataset

# converting Nan values
ds['condition']=ds['condition'].fillna(3)

# hot encoding
#for condition
integer_encoded = ds['condition'].values
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x1 = onehot_encoder.fit_transform(integer_encoded) # 4 columns

#for colortype
integer_encoded = ds['color_type'].values  
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x2 = onehot_encoder.fit_transform(integer_encoded) # 56 columns

#imputation for X1
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

x3 = [0]*len(ds) # 1 column
for i in range(len(ds)):
    if ds['X1'][i] == 0:
        x = int(ds['X2'][i])
        x3[i] = avg[x]
    else:
        x3[i] = ds['X1'][i]

x3 = np.resize(x3, (len(ds), 1))

x_final = np.concatenate( # 66 columns
    (x1,                             # 4
     x2,                             # 56   
     ds[['length(m)']],              # 1
     ds[['height(cm)']],             # 1
     ds[['X1']],                             # 1 
     ds[['X2']],                     # 1
     ds[['breed_category']],         # 1
     ds[['pet_category']]), axis=1)  # 1

final_ds = pd.DataFrame(
    data=x_final[0:,0:],
    index=[i for i in range(x_final.shape[0])],
    columns=['f'+str(i) for i in range(x_final.shape[1])])

final_ds['f62'][final_ds['f62'] == 0] = None

imputer = KNNImputer(n_neighbors=5)
imputer.fit(final_ds)
final_ds = imputer.transform(final_ds)

final_ds = pd.DataFrame(
    data=final_ds[0:,0:],
    index=[i for i in range(final_ds.shape[0])],
    columns=['f'+str(i) for i in range(final_ds.shape[1])])


# data distribution
feature_names = final_ds.columns[0:64]
x_fn = final_ds[feature_names]
# have to find x_fn_breed to predict y_pet
x_fn_breed = pd.concat([x_fn, final_ds['f64']], axis=1)
y_breed = final_ds['f64']
y_pet = final_ds['f65']

# final feature selection
selector = SelectKBest(chi2, k = 64) #chi2, f_classif
feature_x_fn = selector.fit_transform(x_fn, y_breed)
mask = selector.get_support() #list of booleans
feature_x_fn_names = [] # The list of your K best features
for bool, feature in zip(mask, x_fn.columns):
    if bool:
        feature_x_fn_names.append(feature)

#lola = pd.DataFrame(feature_x_fn, feature_x_fn_names)

# use only one
#feature_x_fn_breed = pd.concat([feature_x_fn, final_ds['f64']], axis=1)
#feature_x_fn_breed = SelectKBest(chi2, k=10).fit_transform(x_fn_breed, y_pet)
selector = SelectKBest(chi2, k = 50) #chi2, f_classif
feature_x_fn_breed = selector.fit_transform(x_fn_breed, y_pet)
mask = selector.get_support() #list of booleans
feature_x_fn_breed_names = [] # The list of your K best features
for bool, feature in zip(mask, x_fn_breed.columns):
    if bool:
        feature_x_fn_breed_names.append(feature)



# use classification algorithm for model training
model_1 = GradientBoostingClassifier(random_state=0)
model_1.fit(feature_x_fn, y_breed)
model_2 = GradientBoostingClassifier(random_state=0)
model_2.fit(feature_x_fn_breed, y_pet)


# apply on test data

# load data
td = pd.read_csv("Dataset/test.csv")

# converting Nan values
td['condition']=td['condition'].fillna(3)

# hot encoding
#for condition
integer_encoded = td['condition'].values
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x1 = onehot_encoder.fit_transform(integer_encoded) # 4 columns

#for colortype
integer_encoded = td['color_type'].values  
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
x2 = onehot_encoder.fit_transform(integer_encoded) # 56 columns

#imputation for X1
matrix = [0 for y in range(10)] 
count = [0 for y in range(10)] 
avg = [0 for y in range(10)]

for i in range(len(td)):
    if td['X1'][i] != 0:
        x = int(td['X2'][i])
        matrix[x] = matrix[x] + td['X1'][i]
        count[x] += 1
        
for i in range(10):
    if count[i] != 0:
        avg[i] = round(matrix[i]/count[i])

x3 = [0]*len(td) # 1 column
for i in range(len(td)):
    if td['X1'][i] == 0:
        x = int(td['X2'][i])
        x3[i] = avg[x]
    else:
        x3[i] = td['X1'][i]

x3 = np.resize(x3, (len(td), 1))

x_final = np.concatenate( # 66 columns
    (x1,                             # 4
     x2,                             # 56   
     td[['length(m)']],              # 1
     td[['height(cm)']],             # 1
     td[['X1']],                   # 'X1'    # 1
     td[['X2']]), axis=1)          # 1



final_td = pd.DataFrame(
    data=x_final[0:,0:],
    index=[i for i in range(x_final.shape[0])],
    columns=['f'+str(i) for i in range(x_final.shape[1])])


final_td['f62'][final_td['f62'] == 0] = None

imputer = KNNImputer(n_neighbors = 5)
imputer.fit(final_td)
final_td = imputer.transform(final_td)

final_td = pd.DataFrame(
    data=final_td[0:,0:],
    index=[i for i in range(final_td.shape[0])],
    columns=['f'+str(i) for i in range(final_td.shape[1])])


# to predict the breed
test_df = pd.DataFrame()
for i in feature_x_fn_names:
    test_df[i] = final_td[i]


predict_1 = model_1.predict(test_df)
predict_1 = np.resize(predict_1, (len(td), 1))

# to predict the pet
# modifying content of final_td: adding predict_1 as 'f64'
x_final = np.concatenate( # 66 columns
    (x1,                             # 4
     x2,                             # 56   
     td[['length(m)']],              # 1
     td[['height(cm)']],             # 1
     x3,                   # 'X1'    # 1
     td[['X2']], predict_1), axis=1)          # 1

final_td_2 = pd.DataFrame(
    data=x_final[0:,0:],
    index=[i for i in range(x_final.shape[0])],
    columns=['f'+str(i) for i in range(x_final.shape[1])])
 
test_df = pd.DataFrame()
for i in feature_x_fn_breed_names:
    test_df[i] = final_td_2[i]

predict_2 = model_2.predict(test_df)
predict_2 = np.resize(predict_2, (len(td), 1))

answer = pd.read_csv("lola.csv")
answer['breed_category'] = predict_1
answer['pet_category'] = predict_2
answer.to_csv(r'lola.csv', index=False)
