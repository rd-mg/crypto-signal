import pandas as pd
import numpy as np

from analyzers.informants.lrsi import LRSI
from analyzers.indicators.stoch_rsi import StochasticRSI
from analyzers.utils import IndicatorUtils

def weighted_moving_average(dataframe, period):
    weights = np.arange(1, period + 1)
    wma = dataframe['close'].rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma

def hull_moving_average(dataframe, period=9):
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wmaf = weighted_moving_average(dataframe, half_length)
    wmas = weighted_moving_average(dataframe, sqrt_length)
    hma = weighted_moving_average((2 * wmaf - wmas).to_frame(name='close'), sqrt_length)
    return hma

def stochastic_hma(dataframe, period=9):
    hma = hull_moving_average(dataframe, period)
    lowest_hma = hma.rolling(window=period).min()
    highest_hma = hma.rolling(window=period).max()
    stoch_hma = 100 * (hma - lowest_hma) / (highest_hma - lowest_hma)
    return stoch_hma

class StochHMA(IndicatorUtils):
    def analyze(self, historical_data, signal='stoch_hma', period_count=9, hot_thresh=80, cold_thresh=80):
        dataframe = self.convert_to_dataframe(historical_data)
        stoch_hma_values = stochastic_hma(dataframe, period_count).to_frame(name='stoch_hma')
        stoch_hma_values.dropna(how='all', inplace=True)
        
        # Calculate the previous value of 'stoch_hma' and store in the same DataFrame
        stoch_hma_values['prev_stoch_hma'] = stoch_hma_values['stoch_hma'].shift()

        # Determine hot and cold signals using 'apply' with lambda function
        stoch_hma_values['is_hot'] = stoch_hma_values.apply(
            lambda row: ((row['stoch_hma'] <= hot_thresh) & (row['stoch_hma'] > row['prev_stoch_hma'])) | (row['stoch_hma'] == 0),
            axis=1
        )

        stoch_hma_values['is_cold'] = stoch_hma_values.apply(
            lambda row: (row['stoch_hma'] > cold_thresh) | (row['stoch_hma'] < row['prev_stoch_hma']) | (row['stoch_hma'] == 100),
            axis=1
        )

        # Drop the 'prev_stoch_hma' column if not needed further
        stoch_hma_values.drop('prev_stoch_hma', axis=1, inplace=True)

        return stoch_hma_values



