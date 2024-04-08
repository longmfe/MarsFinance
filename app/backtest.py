import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xtquant import xtdata

#TODO: what if the portfolio inclue equity across the market
market = 'SH'

class Context:
    def __init__(self, market, cash, start_date, end_date):
        self.cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {}
        self.benchmark = None
        self.date_range = xtdata.get_trading_calendar(market, start_date, end_date)

context = Context(1000, 201601, 201701)
print(context.date_range)

