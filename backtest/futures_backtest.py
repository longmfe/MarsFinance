# -*- coding: utf-8 -*-
"""股指期货 CTA 回测引擎：多空双向 + 波动率目标仓位 + 保证金强平。

与 ``stock_backtest.StockBacktest`` 的差异：
- **多空双向**：信号 -1 在空仓时开空（而不是忽略），持仓时信号反向则
  平仓并反手（可关）；
- **合约制**：持仓以手计（整数），每手名义价值 = 价格 × multiplier；
- **波动率目标仓位**：手数 = 现金 × 目标年化波动 / (ATR × multiplier ×
  √252)，同时受保证金上限约束——CTA 按波动率倒推仓位，而不是满保证金
  加杠杆（12% 保证金满仓 ≈ 8 倍杠杆，日常波动就能击穿账户）；
- **保证金强平**：权益低于持仓保证金时按当日收盘价强制平仓（交易所
  追缴保证金的现实约束），账户不会出现负权益；
- **双向佣金**：按名义价值 × commission_rate 收取（股指期货双边收）。

防未来函数与 StockBacktest 一致：第 i 日策略只看到 ``data.iloc[:i]``
（截至前一交易日），成交发生在第 i 日；ATR 同样只用可见数据计算。

账户模型：现金 + 未实现盈亏。保证金视为资金占用上限而非现金流出，
每日权益 = 现金 + 持仓手数 × (现价 - 开仓价) × multiplier。

佣金按单一费率计（未区分平今仓的高费率），研究用途足够。
"""

import numpy as np
import pandas as pd


class FuturesBacktest:
    """单品种期货的事件循环回测器（多空双向，波动率目标仓位）。"""

    def __init__(
        self,
        initial_capital=2_000_000,
        multiplier=300,
        margin_rate=0.12,
        commission_rate=0.000023,
        slippage=0.0005,
        annual_vol_target=0.15,
        atr_period=20,
    ):
        self.initial_capital = initial_capital
        self.multiplier = multiplier
        self.margin_rate = margin_rate
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.annual_vol_target = annual_vol_target
        self.atr_period = atr_period
        self.cash = initial_capital
        self.position = 0  # 有符号手数：正=多，负=空
        self.entry_price = 0.0
        self.open_commission = 0.0
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.margin_calls = 0
        self.current_symbol = None

    def run_backtest(
        self, data, strategy_function, symbol=None, reverse=True, allow_short=True
    ):
        """运行单品种期货回测。

        Args:
            data: 行情 DataFrame，必须含 'close' 列（可含 'date' 列或日期索引；
                波动率目标仓位额外需要 'high'/'low'，缺列时退回保证金上限）。
            strategy_function: 策略函数，输入截至前一日的行情，返回 1/-1/0。
            symbol: 合约代码，仅用于交易记录。
            reverse: 持仓时收到反向信号是否平仓并反手；False 则只平仓。
            allow_short: 是否允许开空（False 即纯多头 CTA，-1 只用于平多）。

        Returns:
            dict: ``calculate_metrics()`` 输出的回测指标。
        """
        self.current_symbol = symbol
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.open_commission = 0.0
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.margin_calls = 0

        for i in range(1, len(data)):
            # 信号只能看到截至前一日的数据——结构性防未来函数
            current_data = data.iloc[:i]
            price = data.iloc[i]["close"]

            if "date" in data.columns:
                current_date = data.iloc[i]["date"]
            else:
                current_date = data.index[i]

            signal = strategy_function(current_data)
            self.execute_trade(signal, price, current_date, current_data, reverse, allow_short)

            equity = (
                self.cash
                + self.position * (price - self.entry_price) * self.multiplier
            )

            # 保证金追缴的现实约束：权益低于持仓保证金时按当日收盘价强制平仓
            margin_required = (
                abs(self.position) * price * self.multiplier * self.margin_rate
            )
            if self.position != 0 and equity < margin_required:
                self._close(price, current_date)
                self.margin_calls += 1
                equity = self.cash

            self.portfolio_values.append(equity)
            self.dates.append(current_date)

        return self.calculate_metrics()

    def execute_trade(self, signal, price, date, current_data=None, reverse=True, allow_short=True):
        """执行信号：1 开多/空翻多，-1 开空/多翻空（allow_short=False 时只平多），0 不动。"""
        if signal == 1:
            if self.position < 0:
                self._close(price, date)
                if reverse:
                    self._open(1, price, date, current_data)
            elif self.position == 0:
                self._open(1, price, date, current_data)
        elif signal == -1:
            if self.position > 0:
                self._close(price, date)
                if reverse and allow_short:
                    self._open(-1, price, date, current_data)
            elif self.position == 0 and allow_short:
                self._open(-1, price, date, current_data)

    def _atr_points(self, data):
        """用截至前一日的行情算 ATR（指数点），无法计算时返回 None。"""
        if "high" not in data.columns or "low" not in data.columns:
            return None

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.tail(self.atr_period).mean()
        if pd.isna(atr) or atr <= 0:
            return None
        return float(atr)

    def _open(self, side, price, date, current_data=None):
        """开仓：side +1 开多 / -1 开空。成交价向不利方向调整。

        手数取两个上限的较小者：
        - **保证金上限**：现金 / (名义价值 × margin_rate)，向下取整；
        - **波动率目标上限**：把组合日波动锁定在 annual_vol_target，
          手数 = 现金 × 目标年化波动 / (ATR × multiplier × √252)。
          这是 CTA 的标准做法——按波动率倒推仓位，而不是满保证金加杠杆。
        """
        fill = price * (1 + self.slippage * side)
        notional_per_contract = fill * self.multiplier

        margin_contracts = int(self.cash / (notional_per_contract * self.margin_rate))

        if self.annual_vol_target is not None:
            atr = self._atr_points(current_data) if current_data is not None else None
            if atr is not None:
                per_contract_daily_vol = atr * self.multiplier
                target_daily_vol = (
                    self.cash * self.annual_vol_target / np.sqrt(252)
                )
                vol_contracts = int(target_daily_vol / per_contract_daily_vol)
                contracts = min(margin_contracts, vol_contracts)
            else:
                contracts = margin_contracts
        else:
            contracts = margin_contracts

        if contracts <= 0:
            return

        commission = contracts * notional_per_contract * self.commission_rate

        self.position = side * contracts
        self.entry_price = fill
        self.open_commission = commission
        self.cash -= commission

        self.trades.append(
            {
                "type": "BUY" if side > 0 else "SELL",
                "side": "LONG" if side > 0 else "SHORT",
                "date": date,
                "price": fill,
                "contracts": contracts,
                "commission": commission,
                "stock": self.current_symbol,
            }
        )

    def _close(self, price, date):
        """平仓：按当前持仓方向反向成交，盈亏计入现金。"""
        side = 1 if self.position > 0 else -1
        fill = price * (1 - self.slippage * side)
        contracts = abs(self.position)
        notional = contracts * fill * self.multiplier

        pnl = self.position * (fill - self.entry_price) * self.multiplier
        commission = notional * self.commission_rate
        profit = pnl - commission - self.open_commission

        self.cash += pnl - commission

        # 平仓信息写回对应的开仓记录
        self.trades[-1].update(
            {
                "exit_date": date,
                "exit_price": fill,
                "profit": profit,
            }
        )

        self.position = 0
        self.entry_price = 0.0
        self.open_commission = 0.0

    def calculate_metrics(self):
        """计算回测指标：总收益、年化、夏普、最大回撤、胜率等。"""
        if len(self.portfolio_values) == 0:
            return {}

        returns = pd.Series(self.portfolio_values).pct_change().dropna()

        if len(returns) == 0:
            return {}

        total_return = (
            self.portfolio_values[-1] - self.initial_capital
        ) / self.initial_capital
        trading_days = len(self.portfolio_values)

        # 亏光本金时 (1 + total_return) 为负，分数次幂无实数解 → 年化取 -100%
        if total_return <= -1:
            annual_return = -1.0
        elif trading_days > 0:
            annual_return = (1 + total_return) ** (252 / trading_days) - 1
        else:
            annual_return = 0

        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        max_drawdown = self.calculate_max_drawdown()

        closed = [t for t in self.trades if "profit" in t]
        winning_trades = len([t for t in closed if t["profit"] > 0])
        win_rate = winning_trades / len(closed) if closed else 0

        return {
            "stock_code": self.current_symbol,
            "initial_capital": self.initial_capital,
            "final_value": (
                self.portfolio_values[-1]
                if self.portfolio_values
                else self.initial_capital
            ),
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(closed),
            "win_rate": win_rate,
            "avg_trade_profit": (
                np.mean([t["profit"] for t in closed]) if closed else 0
            ),
            "long_trades": len([t for t in closed if t.get("side") == "LONG"]),
            "short_trades": len([t for t in closed if t.get("side") == "SHORT"]),
            "margin_calls": self.margin_calls,
        }

    def calculate_max_drawdown(self):
        """计算最大回撤（峰值到谷值的最大跌幅）。"""
        if not self.portfolio_values:
            return 0

        peak = self.portfolio_values[0]
        max_dd = 0

        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd
