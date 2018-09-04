# line plot of time series
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import os


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


pwd='/Users/T/Desktop/ML/StockIndicators/ARIMA/'
os.chdir(pwd)

#Iterate through the dataset:
datasets = os.listdir(pwd+'/Dataset')
n = len(datasets)
#Cleaning up previous Results:
if os.path.exists('Error_Rates_ARIMA.txt'):
    os.remove('Error_Rates_ARIMA.txt')

for i in range(n):

    # set dataset path
    path = pwd+'/Dataset/'+datasets[i]
    # load dataset
    series = Series.from_csv(path, header=0)
    # display first few rows
    split_p = len(series) - 7
    dataset, validation = series[0:split_p], series[split_p:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    dataset.to_csv('dataset.csv')
    validation.to_csv('validation.csv')
    # line plot of dataset
    series.plot()

    plt.show()

    path_d = '/Users/T/Desktop/ML/StockIndicators/ARIMA/dataset.csv'
    series = Series.from_csv(path_d)
    X = series.values
    days_in_year = 2
    differenced = difference(X, days_in_year)
    # fitting
    model = ARIMA(differenced, order=(2,0,0))
    model_fit = model.fit(disp=0)
    # print summ
    # print(model_fit.summary())

    forecast = model_fit.forecast()[0]
    forecast = inverse_difference(X, forecast, days_in_year)
    print('Forecast: %f' % forecast)

    # one-step out of sample forecast
    # start_index = '1990-12-25'
    # end_index = '1990-12-25'
    # forecast = model_fit.predict(start=start_index, end=end_index)

    # Forecast method for future prediction
    forecast = model_fit.forecast(steps=7)[0]

    # Predict method for future prediction
    # forecast = model_fit.predict(start=len(differenced),end=len(differenced)+6)

    history = [x for x in X]
    day = 1
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        print('Day %d: %f' % (day, inverted))
        history.append(inverted)
        day += 1

    # series = pd.read_csv('/Users/T/Desktop/ML/StockIndicators/ARIMA/1/daily-minimum-temperatures-in-me.csv')
    # X = series.values
    # plt.plot(X[-7:,1],color='blue')
    # plt.plot(history[-7:],color='red')
    # plt.show()

    series = pd.read_csv(path, header=0)
    X = series.values
    y1 = history[-7:]
    y = X[-7:, 1]
    err = 0
    plt.plot(y, color='blue',label='Actual')
    plt.plot(y1, color='red',label='Predicted')
    plt.legend(loc='upper left')
    plt.title(datasets[i])
    plt.savefig(pwd + '/Plots/' + datasets[i] + '.png')
    plt.show()
    for i in range(len(y1)):
        err += y[i] - y1[i]
    print("Error : ", err)
    with open ('Error_Rates_ARIMA.txt','a') as f:
        st = "Error Rate for "+datasets[i]+" : "+str(err)
        f.write(st)
        f.close()

