import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.metrics import r2_score,mean_squared_error



data=pd.read_csv('CarPrice_Assignment_One_Var.csv')

#Building data understanding 
print('shape of dataframe is:', data.shape)
data.info()
data['car_ID']=data['car_ID'].astype('object')

data.describe()

#Creating correlation matrix
data_numeric=data.select_dtypes(include=['Float64','int64'])
data_numeric.head()

cor=data_numeric.corr()
cor.round(2)

X=data.loc[:,['horsepower']]
Y=data['price']

plt.scatter(X, Y)
plt.show()

# X = np.array(data['horsepower']).reshape(-1, 1)
# Y = np.array(data['price']).reshape(-1, 1)

#model building
import statsmodels.api as sm
X=sm.add_constant(X)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,random_state=100)
model=sm.OLS(y_train,X_train).fit()

X_train.shape
X_train.head()
print(model.summary())

from statsmodels.tools.eval_measures import rmse
ypred_train=model.predict(X_train)
rmse_train=rmse(y_train,ypred_train)
print(rmse_train)


ypred_train.head()

#Generating the prediction
ypred_test=model.predict(X_test)
rmse_test=rmse(y_test,ypred_test)
print(rmse_test)

Y_max=y_train.max()
Y_min=y_train.min()

import seaborn as sns
ax=sns.scatterplot(x=Y,y=model.fittedvalues)
ax.set(ylim=(Y_min,Y_max))
ax.set(xlim=(Y_min,Y_max))
ax.set_ylabel('predicted value of strenght')
ax.set_xlabel('observed value of strenght')

x_ref=Y_ref=np.linspace(Y_min,Y_max,100)
plt.plot(x_ref,Y_ref,color='red', linewidth=1)
plt.show()







