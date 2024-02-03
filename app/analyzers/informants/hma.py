""" HMA Indicator
"""

import math
import pandas as pd
import numpy as np
import pandas
from talib import abstract

from analyzers.utils import IndicatorUtils

def hull_moving_average(dataframe, period=9):
    """
    Calculate Hull Moving Average (HMA)

    Args:
            dataframe (pd.DataFrame): Input data frame containing the 'close' column.
            period (int): The period over which to calculate HMA.

    Returns:
            pd.Series: A series containing the HMA.
    """
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wmaf = dataframe['close'].rolling(window=half_length).mean()
    wmas = dataframe['close'].rolling(window=period).mean()
    dataframe['hma'] = 2 * wmaf - wmas
    hma = dataframe['hma'].rolling(window=sqrt_length).mean()

    return hma

class HMA(IndicatorUtils):
    def analyze(self, historical_data, period_count=3):
        """Performs an EMA analysis on the historical data

        Args:
        historical_data (list): A matrix of historical OHCLV data.
        period_count (int, optional): Defaults to 15. The number of data points to consider for
                our exponential moving average.

        Returns:
        pandas.DataFrame: A dataframe containing the indicators and hot/cold values.
        """

        dataframe = self.convert_to_dataframe(historical_data)
        hma_values = hull_moving_average(dataframe, period_count).to_frame()
        hma_values.dropna(how='all', inplace=True)
        hma_values.rename(columns={0: 'hma'}, inplace=True)

        return hma_values
