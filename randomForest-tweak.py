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


#converts '%m/%d/%Y' date string to day-of-year number
def date_to_day_of_year(date_str):
    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
    return date_obj.timetuple().tm_yday

#-----------------------------------------------------------------
#read in csv
df = pd.read_csv("input/wastewater/clean_historic.csv") #column names: date, rainfall, average flow, max flow, min flow, fe flow


#drop empty rows
df = df.dropna(ignore_index=True)

#convert dates to day-of-year number
df['day of year'] = df['date'].map(lambda a: date_to_day_of_year(a))

X = df[['day of year', 'rainfall']]  #features
X1 = df['day of year'] 
X2 = df['rainfall']

#select one!
# y = df['average flow']  #target variable
# y = df['max flow']  #target variable
y = df['fe flow']  #target variable


#split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)


#hyperparameter options
random_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}


#set up model with randomized hyperparameters
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs = -1)

#train model
rf_random.fit(X_train,y_train)

# optimal hyperparameters: {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
print(rf_random.best_params_)

#predict results
y_pred = rf_random.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
print('root mean squared error:', rmse)

r2 = r2_score(y_test, y_pred)
print('r-squared:', r2)


#-----------------------------------------------------------------

# plt.title("rainfall (in) vs average flow (mgd)")
# plt.scatter(X2, y, s=5)
# plt.xlabel("rainfall (in)")
# plt.ylabel("average flow (mgd)")
# plt.show()

#-----------------------------------------------------------------
# plt.title("day of year vs average flow (mgd)")
# plt.scatter(X1, y, s=5)
# plt.xlabel("day of year")
# plt.ylabel("average flow (mgd)")
# plt.show()

#-----------------------------------------------------------------

# plt.title("rainfall (in) vs average flow (mgd)")
# plt.scatter(X2, y, s=5)
# plt.xlabel("rainfall (in)")
# plt.ylabel("average flow (mgd)")
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

# plt.title("Model results: rainfall (in) vs fe flow (mgd)")
# plt.scatter(X_test['rainfall'], y_pred, label='predicted flow', s=5)
# plt.scatter(X_test['rainfall'], y_test, label='actual flow', s=5)
# plt.xlabel('rainfall (in)')
# plt.ylabel('fe flow (mgd)')
# plt.legend()
# plt.show()

#-----------------------------------------------------------------

plt.title("Model results: day of year vs fe flow (mgd)")
plt.scatter(X_test['day of year'], y_pred, label='predicted flow', s=5)
plt.scatter(X_test['day of year'], y_test, label='actual flow', s=5)
plt.xlabel('day of year')
plt.ylabel('fe flow (mgd)')
plt.legend()
plt.show()

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
