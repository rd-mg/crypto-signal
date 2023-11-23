""" UpTrend analysis indicator
"""

import numpy
import pandas
from talib import abstract

from analyzers.utils import IndicatorUtils


class UpTrend(IndicatorUtils):
    def analyze(self, key_indicator, key_indicator_index, key_signal, key_period_count=1):
        """ Tests for key_indicator is going uptrend seeing period count back.

        Args:
            key_indicator (pandas.DataFrame): A dataframe containing the results of the analysiscrossover.py
                for the selected key indicator.
            key_signal (str): The name of the key indicator.
            key_indicator_index (int): The configuration index of the key indicator to use.
            key_period_count (integer): how many period analyzer needs to go back to compare current period. Default 1

        Returns:
            pandas.DataFrame: A dataframe containing the indicators and hot/cold values.
        """

        key_indicator_name = '{}_{}'.format(key_signal, key_indicator_index)
        new_key_indicator = key_indicator.copy(deep=True)
        for column in new_key_indicator:
            column_indexed_name = '{}_{}'.format(column, key_indicator_index)
            new_key_indicator.rename(columns={column: column_indexed_name}, inplace=True)

        uptrended_indicator_name = '{}_{}_uptrended'.format(key_signal, key_indicator_index)
        uptrended_indicator_index = key_indicator_index
        new_uptrended_indicator = key_indicator.copy(deep=True)
        new_uptrended_indicator = new_uptrended_indicator.shift(periods=key_period_count)
        for column in new_uptrended_indicator:
            column_indexed_name = '{}_{}_uptrended'.format(column, uptrended_indicator_index)
            new_uptrended_indicator.rename(columns={column: column_indexed_name}, inplace=True)

        combined_data = pandas.concat([new_key_indicator, new_uptrended_indicator], axis=1)
        combined_data.dropna(how='any', inplace=True)

        combined_data['is_hot'] = combined_data[key_indicator_name] > combined_data[uptrended_indicator_name]
        combined_data['is_cold'] = combined_data[key_indicator_name] < combined_data[uptrended_indicator_name]

        return combined_data
