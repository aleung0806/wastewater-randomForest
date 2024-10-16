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

df_outlier = df[['rainfall', 'max flow']]




from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.001) 
iso_forest.fit(df_outlier)
outliers = iso_forest.predict(df_outlier)

X = df[['day of year', 'rainfall']]  #features
X1 = df['day of year'] 
X2 = df['rainfall']

y1 = df['average flow'] 
y2 = df['max flow'] 


def mapColor(a):
    if a:
        return 'blue'
    else:
        return 'orange'

colorMap = [mapColor(xi) for xi in outliers]


colors = ['blue', 'orange']

plt.title("rainfall vs max flow")
plt.scatter(X2, y2, c=outliers, s=10)
plt.show()