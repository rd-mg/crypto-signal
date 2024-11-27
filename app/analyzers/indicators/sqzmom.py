import numpy as np
import pandas as pd

from analyzers.utils import IndicatorUtils


class SQZMOM(IndicatorUtils):
    def analyze(
        self, historical_data, signal=["is_hot"], hot_thresh=None, cold_thresh=None
    ):

        df = self.convert_to_dataframe(historical_data)

        # Parameters setup to match the Pine Script
        length = 20
        mult = 2.0
        length_KC = 20
        mult_KC = 1.5

        # Calculate Bollinger Bands (BB)
        basis = df["close"].rolling(window=length).mean()
        dev = mult * df["close"].rolling(window=length).std(ddof=0)
        df["upper_BB"] = basis + dev
        df["lower_BB"] = basis - dev

        # Calculate Keltner Channel (KC)
        tr = pd.DataFrame(
            {
                "tr0": abs(df["high"] - df["low"]),
                "tr1": abs(df["high"] - df["close"].shift()),
                "tr2": abs(df["low"] - df["close"].shift()),
            }
        ).max(axis=1)
        rangema = tr.rolling(window=length_KC).mean()
        ma = df["close"].rolling(window=length_KC).mean()
        df["upper_KC"] = ma + rangema * mult_KC
        df["lower_KC"] = ma - rangema * mult_KC

        # Squeeze detection
        df["sqzOn"] = (df["lower_BB"] > df["lower_KC"]) & (
            df["upper_BB"] < df["upper_KC"]
        )
        df["sqzOff"] = (df["lower_BB"] < df["lower_KC"]) & (
            df["upper_BB"] > df["upper_KC"]
        )
        df["noSqz"] = ~(df["sqzOn"] | df["sqzOff"])

        # Momentum calculation (val)
        highest = df["high"].rolling(window=length_KC).max()
        lowest = df["low"].rolling(window=length_KC).min()
        m1 = (highest + lowest) / 2
        close_avg = df["close"].rolling(window=length_KC).mean()
        # Adjusted source for linear regression to match Pine Script's val
        source_adj = df["close"] - ((m1 + close_avg) / 2)
        val = source_adj.rolling(window=length_KC).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (length_KC - 1)
            + np.polyfit(np.arange(len(x)), x, 1)[1],
            raw=True,
        )

        # Conditions for entering positions
        # Long condition (squeeze turns off, not in no squeeze condition, and val is positive)
        df["long_cond"] = df["sqzOff"] & ~df["noSqz"] & (val > 0)
        # Short condition (squeeze turns off, not in no squeeze condition, and val is negative)
        df["short_cond"] = df["sqzOff"] & ~df["noSqz"] & (val < 0)

        # Assign long and short signals to the dataframe
        df["is_hot"] = df["long_cond"]
        df["is_cold"] = df["short_cond"]

        return df
