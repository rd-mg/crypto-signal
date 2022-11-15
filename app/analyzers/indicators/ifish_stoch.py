
""" 
IFISH STOCHATIC RSI indicator
"""

from talib import abstract
import numpy as np
import pandas
import pandas_ta as pta

from analyzers.utils import IndicatorUtils


class IFISH_STOCH(IndicatorUtils):

    def analyze(self, historical_data, signal=['ifish_stoch'], hot_thresh=-0.9, cold_thresh=0.0, period_count=5):
        """Check when ifish value cross the Upper/Lower bands.

        Args:
            historical_data (list): A matrix of historical OHCLV data.
            period_count (int, optional): Defaults to 5. 
            signal (list, optional): Defaults ifish_stoch value.
            hot_thresh (float, optional): Defaults to -0.9. The threshold at which this might be
                good to purchase.
            cold_thresh (float, optional): Defaults to 0.9. The threshold at which this might be
                good to sell.            

        Returns:
            pandas.DataFrame: A dataframe containing the indicator and hot/cold values.
        """

        dataframe = self.convert_to_dataframe(historical_data)
                
        df = pandas.DataFrame()
        df = dataframe.copy()
        df.ta.stoch(k=5, smooth_k= 9, append=True)
        df['fastk_t'] = 0.1 * (df['STOCHk_5_3_9'] - 50)
        wma = pandas.DataFrame()
        wma['close'] = df['fastk_t']
        df['fastk_t_avg'] = wma.ta.wma(length= 9, append= True)
        df['ifish_stoch'] = (np.exp(2 * df['fastk_t_avg']) - 1) / (np.exp(2 * df['fastk_t_avg']) + 1)

        # print(df.tail())

        df['is_hot'] = False
        df['is_cold'] = False
        df.dropna(how='all', inplace=True)

        df['is_hot'] = (df['ifish_stoch'] > hot_thresh) & (df['ifish_stoch'] < cold_thresh) & (df['ifish_stoch'] > df['ifish_stoch'].shift(1))
        df['is_cold'] = (df['ifish_stoch'] > cold_thresh) | (df['ifish_stoch'] < df['ifish_stoch'].shift(1))

        return df