import pickle
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("mechcivil.csv") 
x=DataFrame(df,columns=['closing_rank'])
y=DataFrame(df,columns=['InstituteNo'])
x=x.to_numpy()
x=x.reshape(-1,1)
y=y.to_numpy()
y=y.reshape(-1,1)
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=145)
model.fit(x_train,y_train.ravel())
y_predicted=model.predict(x_test)
print(model.score(x_test,y_test))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(x_test,y_predicted)
print(cm)
pickle.dump(model, open('modelmechanical.pkl', 'wb'))
