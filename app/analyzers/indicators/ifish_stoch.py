
""" 
IFISH STOCHATIC RSI indicator
"""

from talib import abstract
import numpy as np
import pandas

from analyzers.utils import IndicatorUtils


class IFISH_STOCH(IndicatorUtils):

    def analyze(self, historical_data, signal=['ifish_stoch'], hot_thresh=-0.9, cold_thresh=0.9, period_count=5):
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
        
        ifish_columns = {
            'fastk': [np.nan] * dataframe.index.shape[0],
            'ifish_stoch': [np.nan] * dataframe.index.shape[0]
        }
        ifish_values = pandas.DataFrame(
            ifish_columns,
            index=dataframe.index
        )
        
        ifish_df_size = ifish_values.shape[0]
        close_data = np.array(dataframe['close'])
        
        if close_data.size > period_count:
            # compute stochrsi
            rsi = abstract.RSI(dataframe, period_count=14)

            stochrsi = (rsi - rsi.rolling(period_count).min()) / (rsi.rolling(period_count).max() - rsi.rolling(period_count).min())
            stochrsi_K = stochrsi.rolling(3).mean()
            stochrsi_D = stochrsi_K.rolling(3).mean()

            kd_values = pandas.DataFrame([stochrsi, stochrsi_K, stochrsi_D]).T.rename(
                columns={0: "stoch_rsi", 1: "slow_k", 2: "slow_d"}).copy()

            kd_values['stoch_rsi'] = kd_values['stoch_rsi'].multiply(100)
            kd_values['slow_k'] = kd_values['slow_k'].multiply(100)
            kd_values['slow_d'] = kd_values['slow_d'].multiply(100)
            # for index in range(period_count, ifish_df_size):
            #     data_index = index - period_count
            #     # ifish_values['fastk'][index] = stoch_rsi['fastk']
            #     # ifish_values['stock_rsi'][index] = stoch_rsi[0][data_index]
            ifish_values['fastk'] = 0.1 * (kd_values['slow_k'] - 50)
            ifish_values['ifish_stoch'] = (
                np.exp(2 * ifish_values['fastk']) - 1) / (np.exp(2 * ifish_values['fastk']) + 1)


        ifish_values['is_hot'] = False
        ifish_values['is_cold'] = False
        ifish_values.dropna(how='all', inplace=True)

        ifish_values['is_hot'] = (ifish_values['ifish_stoch'] > hot_thresh) & (ifish_values['ifish_stoch'] < cold_thresh) & (ifish_values['ifish_stoch'] > ifish_values['ifish_stoch'].shift(1))
        ifish_values['is_cold'] = (ifish_values['ifish_stoch'] > cold_thresh) | (ifish_values['ifish_stoch'] < ifish_values['ifish_stoch'].shift(1))

        return ifish_values