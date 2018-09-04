import sklearn

from dbn import SupervisedDBNClassification
import numpy as np

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=1000,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)



X_train = []
Y_train = []
import math
for _ in range(100):
    X_train.append([1,_,_*_,_*_*_,_*_*_*_,_*_*_*_*_])
    Y_train.append(math.sin(_))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler(copy=True, feature_range=(0, 1))
sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
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