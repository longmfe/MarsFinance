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

def get_today_data(code, fields=['open', 'close', 'high', 'low', 'volume', 'preClose']):
    _period = '1d'
    _count = 1
    code = code+"."+context.market
    today = context.date_range[0]
    xtdata.download_history_data(code, _period, today, today)

    try:
        ret = xtdata.get_market_data_ex(field_list=fields,
                stock_list=[code],
                period=_period,
                start_time='',
                end_time=today,
                count=_count
                )[code]
    except KeyError:
        ret = pd.Series()

    return ret

def _order(code, amount):
    today_data = get_today_data(code)
    if len(today_data) == 0:
        print('no trading today')
        return

    price = today_data['close'].values[0]
    if context.cash - amount * price < 0:
        amount = int(context.cash / price)
        print('not enough cash, adjusted to %d' %(amount))

    if amount % 100 != 0:
        #if sold all you have, not need to adjust
        if amount != -context.positions.get(code, 0):
            amount = int(amount / 100) * 100
            print('adjusted to buyable amount %d' % amount)

    # adjust when selling all you have
    if context.positions.get(code, 0) < -amount:
        amount = context.positions.get(code, 0)
        print('can not sell security exceding the account holding, adjusted to %d' % amount)

    context.positions[code] = context.positions.get(code, 0) + amount
    if context.positions[code] == 0:
        del context.positions[code]

    context.cash -= amount * price


