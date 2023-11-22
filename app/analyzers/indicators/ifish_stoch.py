
""" 
IFISH STOCHATIC RSI indicator
"""

import pandas as pd
import talib
import numpy as np
import pandas_ta as ta

from analyzers.utils import IndicatorUtils


class IFISH_STOCH(IndicatorUtils):
    def analyze(self, historical_data, signal="ifish_stoch", hot_thresh=-0.9, cold_thresh=0.9, period_count=5):
        """
        Calculates the Ifish Stoch indicator using the inverse of the Fisher transform
        :param df: Dataframe with close prices
        :param n: Period for Stochastic oscillator
        :param m: Period for the momentum indicator
        :param r: Harmonic ratio for the Fisher transform
        :return: Dataframe with Ifish Stoch values
        """
        # Calculate Stochastic oscillator
        df = pd.DataFrame()
        df = self.convert_to_dataframe(historical_data)
        # Calcular la transformada de Fisher estocástica
        df.ta.fisher(length= period_count, signal = 9, append= True)
        print(df.tail())

        # Calcular la inversa de la transformada de Fisher estocástica
        df["ifish_stoch"] = np.log((1-df["FISHERT_5_9"])/df["FISHERT_5_9"])
        print(df.tail())

        
        df["is_hot"] = (df["ifish_stoch"] < hot_thresh) & (
            df["ifish_stoch"] > df["ifish_stoch"].shift(1))
        df["is_cold"] = (df["ifish_stoch"] > cold_thresh) | (
            df["ifish_stoch"] < df["ifish_stoch"].shift(1))
        return df
