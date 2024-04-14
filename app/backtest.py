import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xtquant import xtdata
import datetime
import dateutil

#TODO: what if the portfolio inclue equity across the market
market = 'SH'

class Context:
    def __init__(self, market, cash, bkt_start_date, bkt_end_date):
        self.market = market
        self.cash = cash
        self.bkt_start_date = bkt_start_date
        self.bkt_end_date = bkt_end_date
        self.positions = {}
        self.benchmark = None
        self.date_range = xtdata.get_trading_calendar(market, bkt_start_date, bkt_end_date)
        self.bkt_dt = dateutil.parser.parse(bkt_start_date)

context = Context(market, 1000, '20160101', '20170101')
print(context.date_range)

def get_hist_data_by_count(code, count, fields=['open', 'close', 'high', 'low', 'volume', 'preClose']):
    hist_end_date = (context.bkt_dt - datetime.timedelta(days=1)).strftime("%Y%m%d")

    trd_calendar = xtdata.get_trading_calendar(context.market)
    hist_start_date = [dt for dt in trd_calendar if dt <= hist_end_date][-count:][0]

    _period = '1d'
    code = code+"."+context.market
    xtdata.download_history_data(code, _period, hist_start, hist_end)

    ret = xtdata.get_market_data_ex(field_list=fields,
                                    stock_list=[code],
                                    period=_period,
                                    start_time='',
                                    end_time=hist_end_date,
                                    count=count
            )

    return ret

#def get_hist_data_by_count(code, hist_start, hist_end, fields=['open', 'close', 'high', 'low', 'volume', 'preClose']):

