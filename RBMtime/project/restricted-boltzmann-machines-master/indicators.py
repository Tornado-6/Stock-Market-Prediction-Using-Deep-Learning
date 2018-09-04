import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(close, indi, len, period):
    plt.plot(list(range(period,len)),close[period:len], color = 'red')
    plt.plot(list(range(period, len)), indi[period:len], color='blue')
    plt.show()
datapath = '/Users/yash/Documents/machine learning/Tutorial/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/RBMtime/AAPL.csv'


from stockstats import StockDataFrame
stock = StockDataFrame.retype(pd.read_csv(datapath))

