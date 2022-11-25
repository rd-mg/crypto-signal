""" Custom Indicator Increase In  Volume
"""

import numpy as np
from scipy import stats
import pandas_ta as pta


from analyzers.utils import IndicatorUtils


class IIP(IndicatorUtils):
    def analyze(self, historical_data, signal=['iip'], hot_thresh=2, cold_thresh=0, period_count=9):
        """Performs an analysis about the increase in price to detect pump dump on the historical data

        Args:
            historical_data (list): A matrix of historical OHCLV data.
            signal (list, optional): Defaults to iip. The indicator line to check hot against.
            hot_thresh (float, optional): Defaults to 10. 
            cold_thresh: below hot+thresh


        Returns:
            pandas.DataFrame: A dataframe containing the indicator and hot/cold values.
        """

        dataframe = self.convert_to_dataframe(historical_data)
        
        dataframe['candle_length'] = np.abs(dataframe['close']*100000 - dataframe['open']*100000)
        dataframe.ta.zscore(close= dataframe['candle_length'], length=period_count, std=hot_thresh, append=True)
        dataframe['iip'] = np.abs(dataframe[f"ZS_{period_count}"])
        dataframe.dropna(how='all', inplace=True)

        dataframe['is_hot'] = False
        dataframe['is_cold'] = False
        dataframe['is_hot'] = (dataframe["iip"] >= hot_thresh) & (dataframe['close'] > dataframe['open'])
        dataframe['is_cold'] = (dataframe["iip"] >= hot_thresh) & (dataframe['close'] < dataframe['open'])
        
        return dataframe
