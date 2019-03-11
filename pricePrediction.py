import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib import style
style.use('fivethirtyeight')
import datetime
import pandas
import requests
import matplotlib.pyplot as plt
import io

import quandl
ticker =input("input the ticker Symbol? ")
type(ticker)

data = quandl.get("WIKI/"+ticker, start_date="2016-01-01", end_date="2018-01-01", api_key="PYzpDMKuMNJS_v4Ws8Ba")
#data=pd.read_csv("/Users/andizhao/downloads/Google.csv")

data.Close.plot()
plt.xlabel('Date')
plt.ylabel('prices')
plt.show()
data = data.reset_index()
data['Date'] = data['Date'].astype('datetime64[ns]')
data.set_index('Date',inplace=True)



data['Change_Perc']=(data['Adj. Close']-data['Adj. Open'])/data['Adj. Open']*100.0

data=data[['Adj. Open','Adj. Close','Change_Perc','Adj. Volume']]
data.fillna(-9999,inplace=True)


#defining labels and features for regression

#Label
data['Price_After_Month']=data['Adj. Close'].shift(-30)
print("shifted")
print(data)
#Features
X=np.array(data.drop(['Price_After_Month'],1))
X=preprocessing.scale(X)
X=X[:-30]
X_Check=X[-30:]
print(data)
data.dropna(inplace=True)
y=np.array(data['Price_After_Month'])
print("after month")
print(data)
#Splitting the data set for training and testin
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print("training")
print(data)

clf=LinearRegression()
LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
accuracy=accuracy*100
accuracy = float("{0:.2f}".format(accuracy))
print('Accuracy is:',accuracy,'%')


forecast=clf.predict(X_Check)
forecast2=clf.predict(X_test)

last_date=data.iloc[-1].name
mydate = datetime.datetime(2017, 12, 1)
modified_date = mydate+ timedelta(days=1)

date=pd.date_range(mydate,periods=30,freq='D')
df1=pd.DataFrame(forecast,columns=['Forecast'],index=date)
data=data.append(df1)



#to plot the prediction graph
data['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print(data['Forecast'])
data['Forecast'].dropna(inplace=True)

#data3 = pd.DataFrame({'Actual': y_test, 'Predicted': forecast2})
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, forecast2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, forecast2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, forecast2)))
avg=sum(y_test)/len(y_test)
print('mean:', avg )
data2 = quandl.get("WIKI/"+ticker, start_date="2017-12-01", end_date="2017-12-30", api_key="PYzpDMKuMNJS_v4Ws8Ba")
data2['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
percentdiff=[]
firstval=[]

for i in data['Forecast']:
    firstval.append(i);
print(len(firstval))
counter=0
for i in data2['Adj. Close']:
    percentdiff.append(abs(firstval[counter]-i)/i*100)
    counter+=1
data2['Percent difference']=percentdiff
data2['Percent difference'].plot()

plt.xlabel('Date')
plt.ylabel('Percent')
plt.show()




print(percentdiff)
