""" Stochastic RSI Indicator
"""

import math

import numpy
import pandas
from talib import abstract
import pandas_ta as pta


from analyzers.utils import IndicatorUtils


class StochasticRSI(IndicatorUtils):
    def analyze(self, historical_data, period_count=9,
                signal=['stoch_rsi'], hot_thresh=None, cold_thresh=None):
        """Performs a Stochastic RSI analysis on the historical data

        Args:
            historical_data (list): A matrix of historical OHCLV data.
            period_count (int, optional): Defaults to 14. The number of data points to consider for
                our Stochastic RSI.
            signal (list, optional): Defaults to stoch_rsi. The indicator line to check hot/cold
                against.
            hot_thresh (float, optional): Defaults to None. The threshold at which this might be
                good to purchase.
            cold_thresh (float, optional): Defaults to None. The threshold at which this might be
                good to sell.

        Returns:
            pandas.DataFrame: A dataframe containing the indicators and hot/cold values.
        """

        dataframe = self.convert_to_dataframe(historical_data)

        df = pandas.DataFrame()
        df = dataframe.copy()
        df.ta.stochrsi(length=9, rsi_length= 9, k=3, d=3,append=True)

        df['stoch_rsi'] = df['STOCHRSIk_9_9_3_3']
        df['slow_k'] = df['STOCHRSIk_9_9_3_3']
        df['slow_d'] = df['STOCHRSId_9_9_3_3']

        stoch_rsi = pandas.concat([dataframe, df], axis=1)
        stoch_rsi.dropna(how='all', inplace=True)

        if stoch_rsi[signal[0]].shape[0]:
            stoch_rsi['is_hot'] = stoch_rsi[signal[0]] < hot_thresh
            stoch_rsi['is_cold'] = stoch_rsi[signal[0]] > cold_thresh

        return stoch_rsi
