"""Executes the trading strategies and analyzes the results.
"""

import math
from datetime import datetime

import pandas
import structlog
from talib import abstract

from analyzers import *
from analyzers.indicators import *
from analyzers.informants import *


class StrategyAnalyzer():
    """Contains all the methods required for analyzing strategies.
    """

    def __init__(self):
        """Initializes StrategyAnalyzer class """
        self.logger = structlog.get_logger()

    def indicator_dispatcher(self):
        """Returns a dictionary for dynamic anaylsis selector

        Returns:
            dictionary: A dictionary of functions to serve as a dynamic analysis selector.
        """

        dispatcher = {
            'candle_recognition': candle_recognition.Candle_recognition().analyze,
            'aroon_oscillator': aroon_oscillator.Aroon_oscillator().analyze,
            'klinger_oscillator': klinger_oscillator.Klinger_oscillator().analyze,
            'adx': adx.Adx().analyze,
            'ichimoku': ichimoku.Ichimoku().analyze,
            'macd': macd.MACD().analyze,
            'rsi': rsi.RSI().analyze,
            'momentum': momentum.Momentum().analyze,
            'mfi': mfi.MFI().analyze,
            'stoch_rsi': stoch_rsi.StochasticRSI().analyze,
            'stoch_hma_avg_rsi': stoch_hma_avg_rsi.StochHMAAVGRSI().analyze,
            'stoch_hma': stoch_hma.StochHMA().analyze,
            'obv': obv.OBV().analyze,
            'iiv': iiv.IIV().analyze,
            'ma_ribbon': ma_ribbon.MARibbon().analyze,
            'ma_crossover': ma_crossover.MACrossover().analyze,
            'bollinger': bollinger.Bollinger().analyze,
            'bbp': bbp.BBP().analyze,
            'macd_cross': macd_cross.MACDCross().analyze,
            'stochrsi_cross': stochrsi_cross.StochRSICross().analyze,
            'sqzmom': sqzmom.SQZMOM().analyze,
            'natr': natr.NATR().analyze,
            'bollinger_bands': bollinger_bands.Bollinger().analyze,
            'roc': roc.ROC().analyze,
            'ifish_stoch': ifish_stoch.IFISH_STOCH().analyze,
            'iip': iip.IIP().analyze

        }

        return dispatcher

    def informant_dispatcher(self):
        """Returns a dictionary for dynamic informant selector

        Returns:
            dictionary: A dictionary of functions to serve as a dynamic informant selector.
        """

        dispatcher = {
            'sma': sma.SMA().analyze,
            'ema': ema.EMA().analyze,
            'vwap': vwap.VWAP().analyze,
            'ohlcv': ohlcv.OHLCV().analyze,
            'lrsi': lrsi.LRSI().analyze,
            'hma': hma.HMA().analyze,
        }

        return dispatcher

    def crossover_dispatcher(self):
        """Returns a pandas.DataFrame for dynamic crossover selector

        Returns:
            dictionary: A dictionary of functions to serve as a dynamic crossover selector.
        """

        dispatcher = {
            'std_crossover': crossover.CrossOver().analyze
        }

        return dispatcher

    def uptrend_dispatcher(self):
        """Returns a pandas.DataFrame for dynamic uptrend selector

        Returns:
            dictionary: A dictionary of functions to serve as a dynamic uptrend selector.
        """

        dispatcher = {
            'std_uptrend': uptrend.UpTrend().analyze
        }

        return dispatcher
