#predictClass.py
#AUTHOR: Gianpiero Cea, 1425458


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier


#importing the data
probeA = pd.read_csv("../probeA.csv",header=0)
probeB = pd.read_csv("../probeB.csv",header=0)

#utility functions needed for preprocessing phase

# some values are "swapped" betweens columns, so we need to reorder:
def reorder(df):
    # From the initial data exploration it was clear that ther was some corruption in the form of a pemutation 
    #for each of the 4 proteins. This code reorders it.
    copydf=df.copy()
    for letter in ["c","m","n","p"]:
        old_c = copydf[[letter+"1",letter+"2",letter+"3"]]
        c = old_c.values
        c.sort(axis=1)
        c_df = pd.DataFrame(c,columns=old_c.columns)
        copydf[old_c.columns] = c_df
    return copydf

#scale data in standard way
def scale_data(dataFrame):
    df = dataFrame.copy()

    for var in df:
        mean = df[var].mean()
        std = df[var].std()
        assert(std != 0)
        df[var] = (df[var]-mean)/std
    
    return df

#reorder data
probeA = reorder(probeA)
probeB = reorder(probeB)

#we define probeA_data to be the data of probeA with class and tna removed
probeA_data =probeA.drop('tna',1).drop('class',1)
#standardisation
probeA_data_std = scale_data(probeA_data)

#we define probeB_data to be the data of probeA with class and tna removed
probeB_data =probeB
#standardisation
probeB_data_std = scale_data(probeB_data)

tna_target = probeA['tna']
class_target = probeA['class']

#feature expansion
def polynomial_feature_ord(X,n):
    poly = pp.PolynomialFeatures(n)
    out = poly.fit_transform(X)
    feature_names = poly.get_feature_names(X.columns)
    
    X_new = pd.DataFrame(out,columns =feature_names)
    return X_new

#we save variouse polynomial expansions  in case useful

data_ord_2 = polynomial_feature_ord(probeA_data_std,2)
data_ord_3 = polynomial_feature_ord(probeA_data_std,3)
data_ord_4 = polynomial_feature_ord(probeA_data_std,4)

data_ord_2_B = polynomial_feature_ord(probeB_data_std,2)
data_ord_3_B = polynomial_feature_ord(probeB_data_std,3)
data_ord_4_B = polynomial_feature_ord(probeB_data_std,4)



#Feature selection(found by extensive search):
features = ['c3', 'm1^2', 'm1 n3', 'n3^2', 'p1^2']
data = data_ord_2_B[features]

#training set
X_train = data_ord_2.copy()[features]
y_train  = class_target.copy()

#test set
X_test = data

#defines and fit the kNN classifier
m = KNeighborsClassifier(n_neighbors=157,metric='euclidean',weights='distance')
fitted = m.fit(X_train,y_train)

predict = m.predict_proba(X_test[features])[:,1]


df = pd.DataFrame(predict)
df.to_csv("classB.csv", index=False,header = False)
