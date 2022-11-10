""" ROC Indicator
"""

import math

import pandas
from talib import abstract

from analyzers.utils import IndicatorUtils


class ROC(IndicatorUtils):
    def analyze(self, historical_data, signal=['roc'], period_count=14):
        """Performs an ROC analysis on the historical data

                Args:
                        historical_data (list): A matrix of historical OHCLV data.
                        period_count (int, optional): Defaults to 15. The number of data points to consider for
                                our exponential moving average.

                Returns:
                        pandas.DataFrame: A dataframe containing the indicators and hot/cold values.
                """

        dataframe = self.convert_to_dataframe(historical_data)
        roc_values = abstract.ROC(dataframe, period_count).to_frame()
        roc_values.dropna(how='all', inplace=True)
        roc_values.rename(columns={0: 'roc'}, inplace=True)
        dataframe['is_hot'] = True
        dataframe['is_cold'] = False

        return roc_values