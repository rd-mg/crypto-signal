""" NATR Indicator
"""

import math

import pandas
from talib import abstract

from analyzers.utils import IndicatorUtils


class NATR(IndicatorUtils):
    def analyze(self, historical_data, signal=['natr'], hot_thresh=None, cold_thresh=None, period_count=14):
        """Performs an NATR analysis on the historical data

                Args:
                        historical_data (list): A matrix of historical OHCLV data.
                        period_count (int, optional): Defaults to 14. The number of data points to consider for
                                our exponential moving average.

                Returns:
                        pandas.DataFrame: A dataframe containing the indicators and hot/cold values.
                """

        dataframe = self.convert_to_dataframe(historical_data)
        natr_values = abstract.NATR(dataframe, period_count).to_frame()
        natr_values.dropna(how='all', inplace=True)
        natr_values.rename(columns={0: 'natr'}, inplace=True)
        natr_values['is_hot'] = True
        natr_values['is_cold'] = False

        return natr_values