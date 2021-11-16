
"""
@author: cmbbd
"""

import pandas as pd
from matplotlib import pyplot as plt

### Chargement des données

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df1 = df_train

###Preparation des données

pd.to_numeric(df_train.total_night_charge)

mc_churn_no = df1[df1.churn=='no'].total_night_charge      
mc_churn_yes = df1[df1.churn=='yes'].total_night_charge      



## Colonnes contenant 'yes' ou 'no'

yes_no_columns = ['voice_mail_plan','international_plan','churn']
for col in yes_no_columns:
    
    df1[col].replace({'yes': 1,'no': 0},inplace=True)


##Créer des one hot encoder pour les colonnes 'area_code' et 'state'
df2 = pd.get_dummies(data=df1, columns=['area_code','state'])
print(df2.dtypes)

## Mettre toutes les colonnes à la même échelle

cols_to_scale = ["number_vmail_messages","total_day_minutes","total_day_calls","total_day_charge","total_eve_minutes","total_eve_calls","total_eve_charge","total_night_minutes","total_night_calls","total_intl_minutes","total_intl_calls","total_intl_charge",  "number_customer_service_calls"]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

## X données, y: Labels
X = df2.drop('churn',axis='columns')
y = df2['churn']

#Diviser la base de donnée en donnée test et en donnée d'entraînement
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

#Creation du model avec tensorflow
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

#evaluation du modèle

from sklearn.metrics import confusion_matrix , classification_report
yp = model.predict(X_test)

y_pred =[]


for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test,y_pred))