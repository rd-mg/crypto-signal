""" Momentum Indicator
"""

import math

import pandas
from talib import abstract

from analyzers.utils import IndicatorUtils


class Momentum(IndicatorUtils):
    def analyze(
        self,
        historical_data,
        period_count=10,
        signal=["momentum"],
        hot_thresh=None,
        cold_thresh=None,
    ):
        """Performs momentum analysis on the historical data"""

        dataframe = self.convert_to_dataframe(historical_data)
        mom_values = abstract.MOM(dataframe, period_count).to_frame()
        mom_values.dropna(how="all", inplace=True)
        mom_values.rename(columns={mom_values.columns[0]: "momentum"}, inplace=True)

        if mom_values[signal[0]].shape[0] > 0:  # Check if there are any momentum values
            # Compare current momentum to hot threshold and to previous momentum value
            mom_values["is_hot"] = (mom_values[signal[0]] < hot_thresh) & (
                mom_values[signal[0]] > mom_values[signal[0]].shift(1)
            )

            mom_values["is_cold"] = (mom_values[signal[0]] > cold_thresh) & (
                mom_values[signal[0]] < mom_values[signal[0]].shift(1)
            )
        return mom_values
