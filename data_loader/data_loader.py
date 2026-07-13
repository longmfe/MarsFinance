# -*- coding: utf-8 -*-
"""数据加载器：经由 QMT/xtquant 下载沪深300成分股日线行情。

xtquant 随 QMT/MiniQMT 终端分发（不在 PyPI），需从 QMT 安装目录获取。
本模块对 xtdata 采用延迟导入：未安装 xtquant 时不影响包的其余部分使用。
"""

from typing import Dict

import pandas as pd


def _xtdata():
    """延迟导入 xtdata，未安装时给出可操作的报错信息。"""
    try:
        from xtquant import xtdata
    except ImportError as exc:
        raise ImportError(
            "缺少 xtquant：它随 QMT/MiniQMT 终端分发（不在 PyPI），"
            "请从 QMT 安装目录复制到项目或加入 PYTHONPATH。"
        ) from exc
    return xtdata


class DataLoader:
    """行情数据加载器（QMT/xtdata 数据源）。"""

    def __init__(self):
        self.data_sources = {}

    def load_hs300_data(
        self, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """下载沪深300全成分股的日线数据。

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            dict: {股票代码: 行情 DataFrame}
        """
        stock_codes = self.get_hs300_stock_list()
        print(f"获取到 {len(stock_codes)} 只股票")

        stock_data_dict = self.download_hs300_data(
            stock_codes, start_date=start_date, end_date=end_date
        )

        print(f"加载沪深300数据: {start_date} 到 {end_date}")
        return stock_data_dict

    @staticmethod
    def download_hs300_data(stock_codes, start_date="20200101", end_date="20231231"):
        """经 xtdata 下载各股票的日线 OHLCV 数据并整理为字典。"""
        xtdata = _xtdata()
        fields = ["open", "close", "high", "low", "volume", "amount", "preClose"]
        xtdata.download_history_data2(stock_codes, "1d", start_date, end_date)

        stock_data_dict = {}
        for i, code in enumerate(stock_codes):
            print(f"\n正在下载第 {i + 1} 只股票: {code}")
            data = xtdata.get_market_data_ex(
                field_list=fields,
                stock_list=[code],
                start_time=start_date,
                end_time=end_date,
                period="1d",
                count=1000,
            )

            if data and code in data:
                df = data[code]
                stock_data_dict[code] = df
                print(f"成功! 数据形状: {df.shape}, 列名: {df.columns.tolist()}")
            else:
                print("获取失败或数据格式异常")

        return stock_data_dict

    @staticmethod
    def get_hs300_stock_list():
        """获取沪深300成分股代码列表；失败时返回空列表。"""
        xtdata = _xtdata()
        try:
            hs300_constituents = xtdata.get_stock_list_in_sector("沪深300")
            print(f"成功获取 {len(hs300_constituents)} 只沪深300成分股")
            return hs300_constituents
        except Exception as e:
            print(f"获取沪深300成分股列表出错: {e}")
            return []

    def add_data_source(self, name: str, source):
        """注册额外数据源。"""
        self.data_sources[name] = source

    def list_available_data(self) -> list:
        """列出已注册的数据源名称。"""
        return list(self.data_sources.keys())
