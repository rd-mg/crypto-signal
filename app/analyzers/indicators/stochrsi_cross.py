""" MACD Cross
"""

import math
import pandas

from talib import abstract
from analyzers.utils import IndicatorUtils


class StochRSICross(IndicatorUtils):

    def analyze(self, historical_data, period_count=9, signal=['stoch_rsi'], smooth_k = 3, smooth_d = 3, hot_thresh=None, cold_thresh=None):
        """Performs a StochRSI cross analysis on the historical data

        Args:
            historical_data (list): A matrix of historical OHCLV data.
            signal (list, optional): Defaults to macd
            smooth_k (integer): number of periods to calculate the smooth K line
            smooth_d (integer): number of periods to calculate the smooth D line
            hot_thresh (float, optional): Unused for this indicator
            cold_thresh (float, optional): Unused for this indicator            

        Returns:
            pandas.DataFrame: A dataframe containing the indicator and hot/cold values.
        """

        dataframe = self.convert_to_dataframe(historical_data)

        df = pandas.DataFrame()
        df = dataframe.copy()
        df.ta.stochrsi(length=9, rsi_length=9, k=3, d=3, append=True)

        df['stoch_rsi'] = df['STOCHRSIk_9_9_3_3']
        df['smooth_k'] = df['STOCHRSIk_9_9_3_3']
        df['smooth_d'] = df['STOCHRSId_9_9_3_3']

        df.dropna(how='all', inplace=True)
        
        previous_k, previous_d = df.iloc[-2]['smooth_k'], df.iloc[-2]['smooth_d']
        current_k, current_d = df.iloc[-1]['smooth_k'], df.iloc[-1]['smooth_d']

        df['is_hot'] = False
        df['is_cold'] = False

        df.at[df.index[-1], 'is_cold'] = previous_k > previous_d and current_k < current_d
        df.at[df.index[-1], 'is_hot'] = previous_k < previous_d and current_k > current_d

        return df
