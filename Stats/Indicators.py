from pandas import datetime
import pandas as pd
from stockstats import StockDataFrame as sdf


import os

os.chdir('/Users/T/Desktop/ML/StockIndicators/Stats')

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

#Disabling Warnings :
pd.options.mode.chained_assignment = None  # default='warn'

series = pd.read_csv('AAPL.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
split_p = len(series) - 7
dataset, validation = series[0:split_p], series[split_p:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
# dataset.to_csv('dataset.csv')
# validation.to_csv('validation.csv')
X = sdf.retype(dataset)

#SMA - 10 days
dataset['sma'] = X['open_10_sma']
print(dataset['sma'])
#Trend Determination for DBN - SMA
sma_t = [0]*10
for i in range(len(dataset['sma'])):
    if i>9:
        x =  dataset['sma'][i] -  dataset['sma'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        sma_t.append(x)
print(sma_t)
dataset['sma_t'] = sma_t
#EMA
dataset['ema'] = X['open_10_ema']
print(dataset['ema'])
#Trend Determination for DBN - EMA
ema_t = [0]*10
for i in range(len(dataset['ema'])):
    if i>9:
        x =  dataset['ema'][i] -  dataset['ema'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        ema_t.append(x)
print(ema_t)
dataset['ema_t'] = ema_t
#Momentum
dataset['mom'] = X['change']
print(dataset['mom'])
#Trend Determination for DBN - MOM
mom_t = [0]*10
for i in range(len(dataset['mom'])):
    if i>9:
        x =  dataset['mom'][i] - dataset['mom'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        mom_t.append(x)
print(mom_t)
dataset['mom_t'] = mom_t
#Stochastic K
dataset['stk'] = X['kdjk']
print(dataset['stk'])
#Trend Determination for DBN - STK
stk_t = [0]*10
for i in range(len(dataset['stk'])):
    if i > 9:
        x = dataset['stk'][i] - dataset['stk'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        stk_t.append(x)
print(stk_t)
dataset['stk_t'] = stk_t
#Stochastic D
dataset['std'] = X['kdjd']
print(dataset['std'])
#Trend Determination for DBN - STD
std_t = [0]*10
for i in range(len(dataset['std'])):
    if i > 9:
        x = dataset['stk'][i] - dataset['std'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        std_t.append(x)
print(std_t)
dataset['std_t'] = std_t
#RSI
dataset['rsi'] = X['rsi_14']
print(dataset['rsi'])
#Trend Determination for DBN - RSI
rsi_t = [0]*14
for i in range(len(dataset['rsi'])):
    if i > 13:
        x = dataset['rsi'][i]
        if(x > 70):
            x = 0
        elif x < 30:
            x = 1
        else:
            if x > dataset['rsi'][i-1]:
                x = 1
            else:
                x = 0
        rsi_t.append(x)
print(rsi_t)
dataset['rsi_t'] = rsi_t

#MACD
dataset['macd'] = X['macd']
print(dataset['macd'])
#Trend Determination for DBN - MACD
macd_t = [0]*10
for i in range(len(dataset['macd'])):
    if i > 9:
        x = dataset['macd'][i] - dataset['macd'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        macd_t.append(x)
print(macd_t)
dataset['macd_t'] = macd_t

#Williams R
dataset['wr'] = X['wr_10']
print(dataset['wr'])
#Trend Determination for DBN - WR
wr_t = [0]*10
for i in range(len(dataset['wr'])):
    if i > 9:
        x = dataset['wr'][i] - dataset['wr'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        wr_t.append(x)
print(wr_t)
dataset['wr_t'] = wr_t

#Volatility Volume Ratio
dataset['vr'] = X['vr_10']
print(dataset['vr'])
#Trend Determination for DBN - STD
vr_t = [0]*10
for i in range(len(dataset['vr'])):
    if i > 9:
        x = dataset['vr'][i] - dataset['vr'][i-1]
        if(x>0):
            x = 1
        else:
            x = 0
        vr_t.append(x)
print(vr_t)
dataset['vr_t'] = vr_t

#Commodity Channel Index
dataset['cci'] = X['cci_10']
print(dataset['cci'])
#Trend Determination for DBN - CCI
cci_t = [0]*10
for i in range(len(dataset['cci'])):
    if i > 9:
        x = dataset['cci'][i]
        if(x > 150):
            x = 0
        elif x < -150:
            x = 1
        else:
            if x > dataset['cci'][i-1]:
                x = 1
            else:
                x = 0
        cci_t.append(x)
print(cci_t)
dataset['cci_t'] = cci_t


#Deleting extra Cols :
del dataset['rsi_14']
del dataset['close_-1_s']
del dataset['close_-1_d']
del dataset['rs_14']
del dataset['open_10_sma']
del dataset['open_10_ema']
del dataset['change']
del dataset['rsv_9']
del dataset['kdjk_9']
del dataset['kdjk']
del dataset['kdjd_9']
del dataset['kdjd']
del dataset['kdjj_9']
del dataset['kdjj']
del dataset['close_26_ema']
del dataset['close_12_ema']
del dataset['macds']
del dataset['macdh']
del dataset['wr_10']
del dataset['vr_10']
del dataset['middle']
del dataset['middle_10_sma']
del dataset['cci_10']

#Exporting the new Dataset with extracted indicators and trend values
dataset.to_csv('dataset_n.csv')