""" 
HMA of Stock RSI plus Laguerre RSI Indicator
"""

import pandas as pd
import numpy as np

from analyzers.informants.lrsi import LRSI
from analyzers.indicators.stoch_rsi import StochasticRSI

from analyzers.utils import IndicatorUtils


def weighted_moving_average(dataframe, period):
    """
    Calculate Weighted Moving Average (WMA)

    Args:
        dataframe (pd.DataFrame): Input data frame containing the 'close' column.
        period (int): The period over which to calculate WMA.

    Returns:
        pd.Series: A series containing the WMA.
    """
    weights = np.arange(1, period + 1)  # Weighting factors
    wma = dataframe['close'].rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma

def hull_moving_average(dataframe, period=3):
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

    # First WMA for half length
    wmaf = weighted_moving_average(dataframe, half_length)
    
    # Second WMA for full length
    wmas = weighted_moving_average(dataframe, sqrt_length)

    # Calculate the HMA: WMA of (2 * WMA_half - WMA_full) over sqrt_length
    hma = weighted_moving_average((2 * wmaf - wmas).to_frame(name='close'), sqrt_length)

    return hma

class StochHMAAVGRSI(IndicatorUtils):

    def analyze(self, historical_data, signal='stoch_hma_avg_rsi', period_count=9, hot_thresh=50, cold_thresh=50):
        """
        Performs an analysis by combining Hull Moving Average (HMA) of the Stock RSI with the Laguerre RSI (LRSI) to generate trading signals.

        Args:
            historical_data (list): A matrix of historical OHCLV data.
            period_count (int, optional): Defaults to 9. The number of data points to consider for our HMA.
            hot_thresh (float, optional): Defaults to 80. The threshold at which this might be good to purchase.
            cold_thresh (float, optional): Defaults to 100. The threshold at which this might be good to sell.

        Returns:
            pd.DataFrame: A dataframe containing the indicators and hot/cold values.
        """

        # Convert historical data to a pandas DataFrame
        dataframe = self.convert_to_dataframe(historical_data)

        # Initialize and calculate LRSI and StochasticRSI
        lrsi_df = LRSI().analyze(historical_data, period_count)
        stoch_rsi_df = StochasticRSI().analyze(historical_data, period_count)
        
        # Drop NaN values
        lrsi_df.dropna(how='all', inplace=True)
        stoch_rsi_df.dropna(how='all', inplace=True)
        
        # Merge the dataframes
        dataframe = pd.merge(lrsi_df, stoch_rsi_df, left_index=True, right_index=True)

        # Normalize LRSI
        dataframe['lrsi_norm'] = dataframe['lrsi'] * 100

        # Calculate average of LRSI normalized and slow_k, slow_d from StochasticRSI
        avg_rsi = pd.DataFrame()
        avg_rsi['close'] = (dataframe['lrsi_norm'] + dataframe['slow_k'] + dataframe['slow_d']) / 3        
        
        stoch_hma_avg_values = hull_moving_average(avg_rsi)
        stoch_hma_avg_values.dropna(inplace=True)
        stoch_hma_avg_values = stoch_hma_avg_values.to_frame(name='stoch_hma_avg_rsi')
        
        stoch_hma_avg_values['prev_stoch_hma'] = stoch_hma_avg_values['stoch_hma_avg_rsi'].shift()

        # Determine hot and cold signals
        stoch_hma_avg_values['is_hot'] = stoch_hma_avg_values['stoch_hma_avg_rsi'].apply(lambda x: x <= hot_thresh)
        stoch_hma_avg_values['is_cold'] = stoch_hma_avg_values['stoch_hma_avg_rsi'].apply(lambda x: x > cold_thresh)
        
        # Determine hot and cold signals using 'apply' with lambda function
        stoch_hma_avg_values['is_hot'] = stoch_hma_avg_values.apply(
            lambda row: ((row['stoch_hma_avg_rsi'] <= hot_thresh) & (row['stoch_hma_avg_rsi'] > row['prev_stoch_hma'])) | (row['stoch_hma_avg_rsi'] == 0),
            axis=1
        )

        stoch_hma_avg_values['is_cold'] = stoch_hma_avg_values.apply(
            lambda row: (row['stoch_hma_avg_rsi'] > cold_thresh) | (row['stoch_hma_avg_rsi'] < row['prev_stoch_hma']) | (row['stoch_hma_avg_rsi'] == 100),
            axis=1
        )

        return stoch_hma_avg_values
