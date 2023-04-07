import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#Import df

df = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

#EDA

#Checking correlations in df
correlation = df.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'rocket')

plt.figure(figsize = (20,25))

plt.subplot(4,2,1)
plt.gca().set_title('Number of Adults')
sns.countplot(x = 'no_of_adults', palette = 'rocket', data = df)

plt.subplot(4,2,2)
plt.gca().set_title('Number of Children')
sns.countplot(x = 'no_of_children', palette = 'rocket', data = df)

plt.subplot(4,2,3)
plt.gca().set_title('Weekend Nights')
sns.countplot(x = 'no_of_weekend_nights', palette = 'rocket', data = df)

plt.subplot(4,2,4)
plt.gca().set_title('Week Nights')
sns.countplot(x = 'no_of_week_nights', palette = 'rocket', data = df)

plt.subplot(4,2,5)
plt.gca().set_title('Meal Plan')
sns.countplot(x = 'type_of_meal_plan', palette = 'rocket', data = df)

plt.subplot(4,2,6)
plt.gca().set_title('Parking Space')
sns.countplot(x = 'required_car_parking_space', palette = 'rocket', data = df)

plt.subplot(4,2,7)
plt.gca().set_title('Room Type')
sns.countplot(x = 'room_type_reserved', palette = 'rocket', data = df)

plt.subplot(4,2,8)
plt.gca().set_title('Booking Status')
sns.countplot(x = 'booking_status', palette = 'rocket', data = df)

plt.figure(figsize = (20,25))

plt.subplot(3,2,1)
plt.gca().set_title('Arrival Month')
sns.countplot(x = 'arrival_month', palette = 'rocket', data = df)

plt.subplot(3,2,2)
plt.gca().set_title('Arrival Year')
sns.countplot(x = 'arrival_year', palette = 'rocket', data = df)

plt.subplot(3,2,3)
plt.gca().set_title('Segment Type')
sns.countplot(x = 'market_segment_type', palette = 'rocket', data = df)

plt.subplot(3,2,4)
plt.gca().set_title('Repeated Guest')
sns.countplot(x = 'repeated_guest', palette = 'rocket', data = df)

plt.subplot(3,2,5)
plt.gca().set_title('Previous Cancellations')
sns.countplot(x = 'no_of_previous_cancellations', palette = 'rocket', data = df)

plt.subplot(3,2,6)
plt.gca().set_title('Special Requests')
sns.countplot(x = 'no_of_special_requests', palette = 'rocket', data = df)

plt.figure(figsize = (25,20))
sns.set_theme(style="white", palette="Spectral")

#sns.set(color_codes = True)

plt.subplot(2,2,1)
sns.histplot(df['lead_time'], kde = False)

plt.subplot(2,2,2)
sns.histplot(df['arrival_date'], kde = False)

plt.subplot(2,2,3)
sns.histplot(df['avg_price_per_room'], kde = False)

plt.subplot(2,2,4)
sns.histplot(df['no_of_previous_bookings_not_canceled'], kde = False)

#Bivariate Analysis¶: Checking Variables for booking_status:
plt.figure(figsize = (20, 25))
plt.suptitle("Analysis of Variables for Booking Status",fontweight="bold", fontsize=20)

plt.subplot(5,2,1)
sns.countplot(x = 'booking_status', hue = 'no_of_adults', palette = 'rocket', data = df)

plt.subplot(5,2,2)
sns.countplot(x = 'booking_status', hue = 'no_of_children', palette = 'rocket', data = df)

plt.subplot(5,2,3)
sns.countplot(x = 'booking_status', hue = 'no_of_weekend_nights', palette = 'rocket', data = df)

plt.subplot(5,2,4)
sns.countplot(x = 'booking_status', hue = 'market_segment_type', palette = 'rocket', data = df)

plt.subplot(5,2,5)
sns.countplot(x = 'booking_status', hue = 'type_of_meal_plan', palette = 'rocket', data = df)

plt.subplot(5,2,6)
sns.countplot(x = 'booking_status', hue = 'required_car_parking_space', palette = 'rocket', data = df)

plt.subplot(5,2,7)
sns.countplot(x = 'booking_status', hue = 'room_type_reserved', palette = 'rocket', data = df)

plt.subplot(5,2,8)
sns.countplot(x = 'booking_status', hue = 'arrival_year', palette = 'rocket', data = df)

plt.figure(figsize = (20, 25))
plt.suptitle("Analysis Of Variable booking_status",fontweight="bold", fontsize=20)

plt.subplot(5,2,1)
sns.countplot(x = 'booking_status', hue = 'repeated_guest', palette = 'rocket', data = df)

plt.subplot(5,2,2)
sns.countplot(x = 'booking_status', hue = 'no_of_special_requests', palette = 'rocket', data = df)

plt.subplot(5,2,3)
sns.kdeplot(x='lead_time', hue='booking_status', palette = 'rocket', shade=True, data=df)

plt.subplot(5,2,4)
sns.kdeplot(x='arrival_year', hue='booking_status', palette = 'rocket', shade=True, data=df)

plt.subplot(5,2,5)
sns.kdeplot(x='arrival_month', hue='booking_status', palette = 'rocket', shade=True, data=df)

plt.subplot(5,2,6)
sns.kdeplot(x='arrival_date', hue='booking_status', palette = 'rocket', shade=True, data=df)

plt.subplot(5,2,7)
sns.kdeplot(x='avg_price_per_room', hue='booking_status', palette = 'rocket', shade=True, data=df)

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





