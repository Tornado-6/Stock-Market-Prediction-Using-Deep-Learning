import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_float_with_comma(x):
    y = []
    for i in x:
        y.append(float(i.replace(",", "")))
    return y


def convert(str):
    period = 10
    dataset_train = pd.read_csv(str)

    x = dataset_train.iloc[:, 4].values

    try:
        open = dataset_train.iloc[:, 1].values
    except Exception:
        open = read_float_with_comma(dataset_train.iloc[:, 1].values)
    try:
        high = dataset_train.iloc[:, 2].values
    except Exception:
        high = read_float_with_comma(dataset_train.iloc[:, 2].values)
    try:
        low = dataset_train.iloc[:, 3].values
    except Exception:
        low = read_float_with_comma(dataset_train.iloc[:, 3].values)
    try:
        close = dataset_train.iloc[:, 4].values
    except Exception:
        close = read_float_with_comma(dataset_train.iloc[:, 4].values)
    try:
        volume = dataset_train.iloc[:, 5].values
    except Exception:
        volume = read_float_with_comma(dataset_train.iloc[:, 5].values)

    from pyti.simple_moving_average import simple_moving_average
    sma = simple_moving_average(close, period)

    from pyti.weighted_moving_average import weighted_moving_average
    wma = weighted_moving_average(close, period)

    from pyti.momentum import momentum
    mome = momentum(close, period)

    from pyti.relative_strength_index import relative_strength_index
    rsi = relative_strength_index(close, period)

    from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
    macd = moving_average_convergence_divergence(close, short_period=1, long_period=10)

    from pyti.commodity_channel_index import commodity_channel_index
    cci = commodity_channel_index(close, high_data=high, low_data=low, period=period)

    from pyti.williams_percent_r import williams_percent_r
    willr = williams_percent_r(close)

    from pyti.accumulation_distribution import accumulation_distribution
    acd = accumulation_distribution(close_data=close, low_data=low, high_data=high, volume=volume)

    X = []
    Y = []
    for _ in range(10, len(open)):
        tmp = []
        tmp.append(sma[_])
        tmp.append(wma[_])
        tmp.append(mome[_])
        tmp.append(rsi[_])
        tmp.append(macd[_])
        tmp.append(cci[_])
        tmp.append(willr[_])
        X.append(tmp)
        Y.append([close[_]])

    X, Y = np.array(X), np.array(Y)
    return X,Y