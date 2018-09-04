import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_float_with_comma(x):
    y = []
    for i in x:
        y.append(float(i.replace(",", "")))
    return y

period = 10
dataset_train = pd.read_csv('/Users/T/Desktop/ML/StockIndicators/RBMtime/Google_Stock_Price_Train.csv')
dataset_test = pd.read_csv('/Users/T/Desktop/ML/StockIndicators/RBMtime/Google_Stock_Price_Test.csv')
#dataset_train = dataset_test
from  pyti.simple_moving_average import simple_moving_average
x = dataset_train.iloc[:,4].values
ans = simple_moving_average(read_float_with_comma(x),10)

open  = dataset_train.iloc[:,1].values
high = dataset_train.iloc[:,2].values
low = dataset_train.iloc[:,3].values
close = read_float_with_comma(dataset_train.iloc[:,4].values)
volume = read_float_with_comma(dataset_train.iloc[:,5].values)


from  pyti.simple_moving_average import simple_moving_average
sma = simple_moving_average(close,period)

from pyti.weighted_moving_average import weighted_moving_average
wma = weighted_moving_average(close,period)

from pyti.momentum import momentum
mome = momentum(close,period)

from pyti.relative_strength_index import relative_strength_index
rsi = relative_strength_index(close,period)

from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
macd = moving_average_convergence_divergence(close, short_period=1, long_period=10)


from pyti.commodity_channel_index import commodity_channel_index
cci = commodity_channel_index(close,high_data=high, low_data=low, period=period)

from pyti.williams_percent_r import williams_percent_r
willr = williams_percent_r(close)

from pyti.accumulation_distribution import accumulation_distribution
acd = accumulation_distribution(close_data=close,low_data=low, high_data=high, volume=volume)


X = []
Y = []
for _ in range(10,len(open)-1):
    tmp = []
    tmp.append((int)(close[_]>sma[_]))
    tmp.append(wma[_])
    tmp.append(mome[_])
    tmp.append(rsi[_])
    tmp.append(macd[_])
    tmp.append(cci[_])
    tmp.append(willr[_])
    X.append(tmp)
    if close[_+1]>close[_]:
        Y.append(1)
    else:
        Y.append(0)

X, Y = np.array(X), np.array(Y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
Y = sc_y.fit_transform(Y.reshape(-1,1))

#############################################
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
sc = MinMaxScaler(feature_range = (0, 1))
Y = sc.fit_transform(Y)


import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from RBMtime.dbn.tensorflow import SupervisedDBNClassification

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)
Y_test = sc_y.fit_transform(Y_test)


Y_train = Y_train.T
Y_train = Y_train[0,:]
Y_test = Y_test.T
Y_test = Y_test[0,:]

#svm
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, Y_train)

# Predicting a new result
Y_pred = regressor.predict(X_test)
Y_pred =sc_y.inverse_transform(Y_pred);


plt.plot(list(range(25)),Y_pred[:25],color = 'red')
plt.plot(list(range(25)),Y_test[:25],color = 'green')
plt.show()

error = 0.0;
for x in range(len(Y_pred)):
    error += abs(Y_pred[x] - Y_test[x])/Y_test[x];
error = error*100/len(Y_test);
acc =  100-error;
print("acc :" , acc)

#random forest

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
Y_test = Y_test.T[0]
error = 0.0;
for x in range(len(Y_pred)):
    tmp = Y_pred[x]
    if tmp > 0.5:
        tmp = 0;
        Y_pred[x] =0
    else :
        tmp = 0;
        Y_pred[x] = 1;
    if tmp is not (Y_test[x]):
        error +=1
error = error*100/len(Y_test);
acc =  100-error;
print("acc :" , acc)



#DBM
# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)


Y_pred = np.array(classifier.predict(X_train))

error = 0;
for i in range(len(Y_train)):
    try :
        error += abs(Y_train[i]-Y_pred[i])/abs(Y_train[i]);
    except Exception:
        error+= 0
error = error/len(Y_train)
print(1-error)