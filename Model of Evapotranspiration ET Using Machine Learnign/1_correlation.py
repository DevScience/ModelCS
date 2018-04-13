# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:31:27 2018
Calculate ET
@author: Jose
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import confusion_matrix

# Load the data from the ET
import pandas as pd
et_data = pd.read_csv("EToCor.csv")

#X = et_data.iloc[:,1:8].values
#y = et_data.iloc[:,8].values

# Compute the correlation
import pandas as pd
correlation = et_data.corr(method='pearson')

import numpy as np; np.random.seed(42)
import seaborn as sns; sns.set()
ax = sns.heatmap(correlation)
# Compute variance
variance = np.var(et_data)

#==============================================================================
# # Summary Statistics
# df = pd.DataFrame({'object': ['a', 'b', 'c'],
#                    'numeric': [1, 2, 3],
#                    'categorical': pd.Categorical(['d','e','f'])
#                    })
#==============================================================================
# 
# Summary Statistics
summaryStat = et_data.describe()


# Classification
# Clustering
# Regretion
# Neural Net 
# Deep Learning

#==============================================================================
# # Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
#==============================================================================

#==============================================================================
# # Encoding categorical data
# # Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# # Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
#==============================================================================

#==============================================================================
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#==============================================================================

#==============================================================================
# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""
#==============================================================================



#==============================================================================
# #Considereaciones Finales pdf FAO
# no se tendría que estar ecuacionando
# o generando formulas o metodos para cada zona
# solo basta entrenar el modelo y generar los comportamiendos
# a partir de una base de datos
# 
# # Avaliar los resultados obtenidos por los modelos comparandolos con un sistema real (tanque clase-A)
# # Sería muy bueno NO CONSIDERAR ETo Calculada para el entrenamiento, ES MEJOR CONSIDERAR ETo real
# # Así se descarta errores de los modelos anteriores.
# # Hargreaves vs FAO CROPWAT vs AI
# # Como podría a una determinada área que no posee datos?
# # Clasificación climática de Köppen
#==============================================================================