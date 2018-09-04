from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
import numpy as np

import os

os.chdir('/Users/T/Desktop/ML/StockIndicators/ARIMA')

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')
#'190' +


series = read_csv('AAPL_2016.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
#
# import math
# def log(X):
#     x = []
#     for _ in X:
#         x.append(math.log(_))
#     x = np.array(x)
#     return x
#
# X = np.array(X)
# X_pre = log(X[:250])/math.log(10)
# X_next = log(X[1:])/math.log(10)
# X_diff = X_next-X_pre
#
# plt.plot(list(range(250)),X_diff)
# plt.show()

# X = X_diff
autocorrelation_plot(X)
plt.show()
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
model = ARIMA(history, order=(5, 1, 0))
model_fit = model.fit(disp=0)
output = model_fit.predict(start=1,end=10)
yhat = output
# predictions.append(yhat)
# obs = test[t]
# history.append(obs)
print('predicted=%f, expected=%f' % (yhat, test))#
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(np.array(predictions)-test, color='red')
plt.show()