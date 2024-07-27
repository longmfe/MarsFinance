import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xtquant import xtdata
import datetime
import dateutil

class Context:
    def __init__(self, market, cash, bkt_start_date, bkt_end_date):
        self.market = market
        self.cash = cash
        self.bkt_start_date = bkt_start_date
        self.bkt_end_date = bkt_end_date
        self.positions = {}
        self.benchmark = None
        self.date_range = xtdata.get_trading_calendar(market, bkt_start_date, bkt_end_date)
        self.bkt_dt = None

class G:
    pass

def set_benchmark(code):
    context.benchmark = code

def get_hist_data_by_count(context, code, count, dividend_type='none', period='1d', fields=['open', 'close', 'high', 'low', 'volume', 'amount', 'preClose']):
    hist_end_date = (dateutil.parser.parse(context.bkt_dt) - datetime.timedelta(days=1)).strftime("%Y%m%d")

    trd_calendar = xtdata.get_trading_calendar(context.market)
    hist_start_date = [dt for dt in trd_calendar if dt <= hist_end_date][-count:][0]

    
    code = code+"."+context.market
    xtdata.download_history_data(code, period, hist_start_date, hist_end_date)

    ret = xtdata.get_market_data_ex(field_list=fields,
                                    stock_list=[code],
                                    period=period,
                                    start_time='',
                                    end_time=hist_end_date,
                                    count=count,
                                    dividend_type=dividend_type
            )[code]

    return ret

def get_hist_data_by_daterange(code, hist_start_date, hist_end_date, fields=['open', 'close', 'high', 'low', 'volume', 'amount', 'preClose']):
    _period = '1d'
    code = code+"."+context.market
    xtdata.download_history_data(code, _period, hist_start_date, hist_end_date)

    ret = xtdata.get_market_data_ex(field_list=fields,
                                    stock_list=[code],
                                    period=_period,
                                    start_time=hist_start_date,
                                    end_time=hist_end_date,
                                    count=-1
            )[code]

    return ret

def get_today_data(code, fields=['open', 'close', 'high', 'low', 'volume', 'amount', 'preClose']):
    _period = '1d'
    _count = 1
    code = code+"."+context.market
    today = context.bkt_dt
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
    print('trading amount %d' % amount)
    today_data = get_today_data(code)
    if len(today_data) == 0:
        print('no trading today')
        return

    price = today_data['open'].values[0]
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

def order(code, amount):
    _order(code, amount)

# buy stock to target amount
def order_target_amount(code, target):
    if target < 0:
        print("target should not be less than 0")
        target = 0

    #TODO: considering T+1 trade: closable amount & total amount
    hold_amount = context.positions.get(code, 0)
    delta_amount = target - hold_amount
    _order(code, delta_amount)

# buy stock of specific value
def order_value(code, value):
    today_data = get_today_data(code)
    amount = int(value / today_data['open'].values[0])
    _order(code, amount)

# buy stock to target value
def order_target_value(code, target):
    if target < 0:
        print("target should not be less than 0")
        target = 0

    today_data = get_today_data(code)
    price = today_data['open'].values[0]
    hold_value = context.positions.get(code, 0) * price

    delta_value = target - hold_value
    order_value(code, delta_value)


def run():
    initialize(context)
    plt_df = pd.DataFrame(index=context.date_range, columns=['price','value'])
    init_value = context.cash
    for dt in context.date_range:
        context.bkt_dt = dt
        handle_data(context)
        
        value = context.cash
        last_price = {}
        for stock in context.positions:
            today_data = get_today_data(stock)
            # stock suspended
            if len(today_data) == 0:
                # if stock suspended on the first day, you won't have it in your context.positions
                price = last_price[stock]
            else:
                price = today_data['open'].values[0]
                last_price[stock] = price
            value += price * context.positions[stock]
        plt_df.loc[dt, 'price'] = price
        plt_df.loc[dt, 'value'] = value
    plt_df['ROE'] = (plt_df['value'] - init_value) / init_value
    bm_df = get_hist_data_by_daterange(context.benchmark, context.bkt_start_date, context.bkt_end_date)
    bm_init = bm_df['open'][0]
    plt_df['benchmark'] = (bm_df['open'] - bm_init) / bm_init

    plt_df[['ROE', 'benchmark']].plot()
    plt.show()
    plt_df.to_csv('plt_df.csv', index=True)


def initialize(context):
    set_benchmark('000001')
    g.p1 = 5
    g.p2 = 60
    g.code = '601318'

def handle_data(context):
    hist_data = get_hist_data_by_count(context, g.code, g.p2)
    ma5 = hist_data['open'][-5:].mean()
    ma60 = hist_data['open'].mean()

    if ma5 > ma60 and g.code not in context.positions:
        order_value(g.code, context.cash)
    elif ma5 < ma60 and g.code in context.positions:
        order_target_amount(g.code, 0)

if __name__ == '__main__':
    #TODO: what if the portfolio inclue equity across the market
    market = 'SH'
    global context
    global g
    context = Context(market, 100000000, '20140101', '20240101')
    run()
