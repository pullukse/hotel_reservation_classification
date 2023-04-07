import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

#Checking correlations in df
correlation = df.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'rocket')


#Model creation

train = pd.get_dummies(df)
test = pd.get_dummies(test)

train = train.drop(['type_of_meal_plan_Meal Plan 3'], axis=1)

y_train = train.pop('booking_status_Canceled')
y_test = test.pop('booking_status_Canceled')

model_lgb = lgb.LGBMClassifier(max_depth = 8, learning_rate = .05, random_state=12349)
model_lgb.fit(train,y_train)

#Model training accuracy
ea = model_lgb.score(train, y_train)

#Model pred
y_pred = model_lgb.predict(test)

#Model test accuracy
a = accuracy_score(model_lgb.predict(test), y_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred))





