import pickle
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("Book2.csv")
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
model=RandomForestClassifier(n_estimators=80)
model.fit(x_train,y_train.ravel())
y_predicted=model.predict(x_test)
print(model.score(x_test,y_test))



# from sklearn import metrics
# #define metrics
# y_pred_proba = model.predict_proba(x_test)[::,1]
# y_pred_proba11 = model.predict_proba(x_test)
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label=1)
# auc = metrics.roc_auc_score(y_test, y_pred_proba11, multi_class="ovr")

# #create ROC(Receiver Operating Characteristics) curve for Random Forest Classifier
# plt.plot(fpr,tpr,label="AUC="+str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()