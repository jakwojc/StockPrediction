
#https://www.researchgate.net/publication/309492895_Forecasting_to_Classification_Predicting_the_direction_of_stock_market_price_using_Xtreme_Gradient_Boosting

import pandas as pd
import collections
import math
import numpy as np



### RSI = 100 - 100/(1+RS)
### RS = avg gain over 14 days/avg loss over 14 days

def RSI(values):
    """Relative strength index(RSI). The RSI is classiﬁed as a momentum oscillator, measuring
the velocity and magnitude of directional price movements. 
The RSI is most typically used on a 14-day timeframe, mea-
sured on a scale from 0 to 100, with high and low levels
marked at 70 and 30, respectively. Shorter or longer time-
frames are used for alternately shorter or longer outlooks.
More extreme high and low levels-80 and 20, or 90 and 10 -
occur less frequently but indicate stronger momentum.

RSI = 100 - 100/(1+RS)
RS = avg gain over 14 days/avg loss over 14 days

    Args:
        values (array or list?): prices time series

    Returns:
        array: RSI values for the input series
    """
    RSI = []
    # values = df.values
    gains = collections.deque(maxlen=14)
    losses = collections.deque(maxlen=14)
    for i in range(len(values)):
        if (i>1):
            diff = values[i] - values[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(diff)
        if (i<15):
            RSI.append(0)
            continue
        ### RSI = 100 - 100/(1+RS)
        ### RS = avg gain over 14 days/avg loss over 14 days
        gains_sum = sum(gains)
        loss_sum = abs(sum(losses))
        RS = gains_sum/loss_sum
        print(RS)
        RSI.append(100 - 100/(1+RS))
    return RSI
    



### %k = 100 * (C-L)/(H-L)
### H = highest price over 14 days
### L = lowesr price over 14 days
### C = current price
def proc_k(values):
    """ Stochastic oscillator
This method attempts to predict price 
turning points by comparing the
closing price of a security to its price range

%k = 100 * (C-L)/(H-L)
H = highest price over 14 days
L = lowesr price over 14 days
C = current price

    Args:
        values (array or list?): prices time series

    Returns:
        array: %k values for the input series
    """
    proc_k = []
    prices = collections.deque(maxlen=14)
    for i in range(len(values)):
        prices.append(values[i])
        if (i<14):
            proc_k.append(0)
            continue
        ### RSI = 100 - 100/(1+RS)
        ### RS = avg gain over 14 days/avg loss over 14 days
        highest = max(prices)[0]
        lowest = min(prices)[0]
        current = values[i][0]
        res = 100 * (current - lowest)/(highest-lowest)
        proc_k.append(res)
    return proc_k

    
### William % R
def proc_r(values):
    """Williams %R ranges from -100 to 0. When its value is above
-20, it indicates a sell signal and when its value is below -80,
it indicates a buy signal.

R = -100 * (H-C)/(H-L)

C - current price
L - lowest price over 14 days
H - highest price over 14 days

    Args:
        values (array or list?): prices time series

    Returns:
        array: William % R values for the input series
    """
    R = []
    prices = collections.deque(maxlen=14)
    for i in range(len(values)):
        prices.append(values[i])
        if (i<14):
            R.append(0)
            continue
        
        highest = max(prices)[0]
        lowest = min(prices)[0]
        current = values[i][0]
        res = -100 * (highest - current)/(highest - lowest)
        R.append(res)
    return R

class EMA():
    def __init__(self, alpha = 0.5, window_len = 10):
        self.value = 0
        self._contents = collections.deque(maxlen=window_len)
        self._amount = 0
        self._window_len = window_len
        self._alpha = alpha
        self._coeff = self._alpha * math.pow(1-self._alpha,self._window_len)
    
    def add(self, values : list):
        for value in values:
            self._add_one(value)   
    def _add_one(self,x):
        if self._amount == 0:
            self.value = x
        else:
            self.value = self._alpha * x + (1 - self._alpha) * self.value
        
        self._amount = self._amount + 1
        if self._amount == self._window_len + 1:
            self.value = self.value - self._coeff/self._alpha * self._contents.pop()
        elif self._amount > self._window_len + 1:
            self.value = self.value - self._coeff * self._contents.pop()
        self._contents.append(x)

def MACD(values):
    """Moving Average Convergence Divergence (MACD)
    When the MACD goes below the SignalLine, it indicates a
sell signal. When it goes above the SignalLine, it indicates
a buy signal.

MACD = EM_12(C) - EM_26(C)
SignalLine = EM_9(MACD)

EM_N - exponential moving average over n days

    Args:
        values (array or list?): prices time series

    Returns:
        array: MACD values for the input series
        array: signal line values
    """
    MACD = []
    signal_line = []
    ema12 = EMA(window_len=12)
    ema26 = EMA(window_len=26)
    ema9 = EMA(window_len=9)
    for curr in values:
        ema12.add([curr])
        ema26.add([curr])
        MACD.append(ema12.value - ema26.value)
        ema9.add([MACD[-1]])
        signal_line.append(ema9.value)
    return MACD, signal_line