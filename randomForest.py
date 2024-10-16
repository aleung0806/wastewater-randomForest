import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest


#-----------------------------------------------------------------
#read in csv
df = pd.read_csv("input/wastewater/clean_historic.csv") #column names: date, rainfall, average flow, max flow, min flow, fe flow


#drop empty rows
df = df.dropna(ignore_index=True)

#convert dates and and day-of-year column
df['datetime'] = pd.to_datetime(df['date'])
df['day of year'] = df['datetime'].dt.dayofyear

df = df.sort_values(by='datetime')


#adds "past average flow" column
#-----------------------------------------------------------------

# num_days = 3
# past_averages = []
# for index, row in df.iterrows():
#     if len(past_averages) > num_days:
#         past_averages.pop(0)
#         df.at[index, 'past average flow'] = sum(past_averages) / num_days
#     else: 
#         df.at[index, 'past average flow'] = row['average flow']
#     past_averages.append(row['average flow'])

# print(df)


#define features
#-----------------------------------------------------------------
X = df[['day of year', 'rainfall']]  #features
X1 = df['day of year'] 
X2 = df['rainfall']

y1 = df['average flow'] 
y2 = df['max flow'] 

#select a target variable
y = df['average flow']
# y = df['max flow']  
# y = df[['max flow', 'average flow']]  


#split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)


# to locate specific records
# outlier = df.loc[df['rainfall'] > 2 ]

outliers = [117, 1171]
X_train = X_train.drop([i for i in outliers if i in X_train.index])
y_train = y_train.drop([i for i in outliers if i in y_train.index])


# random hyperparameters
# random_grid = {
#     'bootstrap': [True, False],
#     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'max_features': ['log2', 'sqrt'],
#     'min_samples_leaf': [1, 2, 4],
#     'min_samples_split': [2, 5, 10],
#     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# }
# regressor = RandomForestRegressor()
# rf = RandomizedSearchCV(estimator=regressor, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs = -1)

#multi output
# rf = RandomForestRegressor(n_estimators=1400, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', max_depth=10, bootstrap=True)

# single output (average flow)
rf = RandomForestRegressor(n_estimators=2000, min_samples_split=10, min_samples_leaf=2, max_features='sqrt', max_depth=10, bootstrap=True)

# single output (max flow)
# rf = RandomForestRegressor(n_estimators=1400, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', max_depth=10, bootstrap=True)


#train model
rf.fit(X_train, y_train)
# print('best params', rf.best_params_)

#predict results
y_pred = rf.predict(X_test)

#formatting for multiple output
# y_pred = {
#     'max flow': results[:,0],
#     'average flow' : results[:,1]
# }

rmse = root_mean_squared_error(y_test, y_pred)
print('root mean squared error:', rmse)

r2 = r2_score(y_test, y_pred)
print('r-squared:', r2)


# plt.title("average flow vs max flow (mgd)")
# plt.scatter(y1, y2, s=5)
# plt.xlabel("average flow (mgd)")
# plt.ylabel("max flow (mgd)")
# plt.show()

#-----------------------------------------------------------------
# plt.title("average flow vs past average flow")
# plt.scatter(df['datetime'], df['average flow'], label='average flow',  s=10)
# plt.scatter(df['datetime'], df['past average flow'], label='past average flow', s=5)
# plt.xlabel("average flow")
# plt.ylabel("past average flow")
# plt.legend()
# plt.show()

#-----------------------------------------------------------------
# plt.title("day of year vs average flow (mgd)")
# plt.scatter(X1, y1, s=5)
# plt.xlabel("day of year")
# plt.ylabel("average flow (mgd)")
# plt.show()

#-----------------------------------------------------------------

# plt.title("rainfall (in) vs average flow (mgd)")
# plt.scatter(X2, y1, s=5)
# plt.xlabel("rainfall (in)")
# plt.ylabel("average flow (mgd)")
# plt.show()

#-----------------------------------------------------------------
# plt.title("day of year vs max flow (mgd)")
# plt.scatter(X1, y2, s=5)
# plt.xlabel("day of year")
# plt.ylabel("max flow (mgd)")
# plt.show()

#-----------------------------------------------------------------
# plt.title("rainfall (in) vs max flow (mgd)")
# plt.scatter(X2, y2, s=5)
# plt.xlabel("rainfall (in)")
# plt.ylabel("max flow (mgd)")
# plt.show()

#-----------------------------------------------------------------
# plt.title("Model results: rainfall (in) vs average flow (mgd)")
# plt.scatter(X_test['rainfall'], y_pred, label='predicted flow', s=5)
# plt.scatter(X_test['rainfall'], y_test, label='actual flow', s=5)
# plt.xlabel('rainfall (in)')
# plt.ylabel('average flow (mgd)')
# plt.legend()
# plt.show()

#-----------------------------------------------------------------

# plt.title("Model results: day of year vs average flow (mgd)")
# plt.scatter(X_test['day of year'], y_pred, label='predicted flow', s=5)
# plt.scatter(X_test['day of year'], y_test, label='actual flow', s=5)
# plt.xlabel('day of year')
# plt.ylabel('average flow (mgd)')
# plt.legend()
# plt.show()

#-----------------------------------------------------------------
# plt.title("Model results: rainfall (in) vs max flow (mgd)")
# plt.scatter(X_test['rainfall'], y_pred, label='predicted flow', s=5)
# plt.scatter(X_test['rainfall'], y_test, label='actual flow', s=5)
# plt.xlabel('rainfall (in)')
# plt.ylabel('max flow (mgd)')
# plt.legend()
# plt.show()

#-----------------------------------------------------------------

# plt.title("Model results: day of year vs max flow (mgd)")
# plt.scatter(X_test['day of year'], y_pred, label='predicted flow', s=5)
# plt.scatter(X_test['day of year'], y_test, label='actual flow', s=5)
# plt.xlabel('day of year')
# plt.ylabel('max flow (mgd)')
# plt.legend()
# plt.show()


#-----------------------------------------------------------------
# #3D scatter
# fig = plt.figure()

# # Create 3D axes
# ax = fig.add_subplot(111, projection='3d')

# #scatter
# sc = ax.scatter(X1, X2, y, s=5, marker='o',color='red')


# #lines
# z = np.ones(shape=X1.shape) * min(y)
# for i,j,k,h in zip(X1,X2,y,z):
#     ax.plot([i,i],[j,j],[k,h], color='red')

# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('y')

# plt.show()

#-----------------------------------------------------------------
#contour map

# colors = ["#8B0000", "#e93e3a", "#ed683c", "#f3903f", "#fdc70c", "#fff33b"]
# values = [50, 40, 30, 20, 10, 0]

# def value_to_color(value):
#     for i in range(6):
#         if value > values[i]:
#             return colors[i]
        
# colorMap = y.map(lambda a: value_to_color(a))

# plt.title("rainfall (in) and day of year vs average flow (mgd)")
# plt.scatter(X1, X2, color=colorMap, s=10)
# plt.xlabel('day of year')
# plt.ylabel('rainfall (in)')
# plt.show()
