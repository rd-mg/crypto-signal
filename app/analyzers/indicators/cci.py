import pandas as pd
from talib import abstract

from analyzers.utils import IndicatorUtils


class CCI(IndicatorUtils):
    def analyze(
        self,
        historical_data,
        period_count=9,
        signal=["cci"],
        hot_thresh=-100,
        cold_thresh=100,
    ):
        """Performs CCI analysis on the historical data

        Args:
            historical_data (list): A matrix of historical OHLCV data.
            period_count (int, optional): The number of data points to consider for the CCI calculation. Defaults to 9.
            hot_thresh (int, optional): The threshold for identifying overbought conditions. Defaults to 100.
            cold_thresh (int, optional): The threshold for identifying oversold conditions. Defaults to -100.

        Returns:
            pandas.DataFrame: A dataframe containing the cci values and hot/cold signal flags.
        """

        dataframe = self.convert_to_dataframe(historical_data)
        cci_values = abstract.CCI(dataframe, timeperiod=period_count).to_frame()
        cci_values.dropna(how="all", inplace=True)
        cci_values.rename(columns={cci_values.columns[0]: "cci"}, inplace=True)

        # Identify hot (overbought) and cold (oversold) signals
        cci_values["is_hot"] = (cci_values[signal[0]] <= hot_thresh) | (
            cci_values[signal[0]] > cci_values[signal[0]].shift()
        )
        cci_values["is_cold"] = (cci_values[signal[0]] >= cold_thresh) | (
            cci_values[signal[0]] < cci_values[signal[0]].shift()
        )

        return cci_values
