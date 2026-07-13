import time
from typing import Dict, List, Tuple

import pandas as pd
from xtquant import xtdata


class AdvancedDataLoader:
    """高级数据加载器，支持下载状态监控"""

    def __init__(self, timeout=300, check_interval=5):
        self.timeout = timeout
        self.check_interval = check_interval
        self.download_status = {}

    def download_with_guarantee(
        self, stock_codes: List[str], start_date: str, end_date: str, period: str = "1d"
    ) -> Tuple[bool, List[str]]:
        """
        保证式下载：确保数据下载完成后再返回
        返回: (是否全部完成, 已完成的股票列表)
        """

        print(f"🔽 开始保证式下载: {len(stock_codes)} 只股票")

        # 初始化下载状态
        for stock_code in stock_codes:
            self.download_status[stock_code] = {
                "requested": False,
                "completed": False,
                "data_available": False,
            }

        # 启动下载
        requested_stocks = []
        for stock_code in stock_codes:
            try:
                success = xtdata.download_history_data2(
                    stock_list=[stock_code],
                    period=period,
                    start_time=start_date,
                    end_time=end_date,
                )

                self.download_status[stock_code]["requested"] = True
                requested_stocks.append(stock_code)

                if success:
                    print(f"✅ {stock_code}: 下载请求成功")
                else:
                    print(f"⚠️ {stock_code}: 下载请求返回失败")

            except Exception as e:
                print(f"❌ {stock_code}: 下载请求异常 - {e}")

        # 监控下载进度
        return self._monitor_download_progress(
            requested_stocks, start_date, end_date, period
        )

    def _monitor_download_progress(
        self, stock_codes: List[str], start_date: str, end_date: str, period: str
    ) -> Tuple[bool, List[str]]:
        """监控下载进度"""

        start_time = time.time()
        completed_stocks = []

        print("📊 开始监控下载进度...")

        while time.time() - start_time < self.timeout:
            current_completed = []

            for stock_code in stock_codes:
                if stock_code in completed_stocks:
                    continue

                # 检查数据是否可用
                data_available = self._check_data_available(
                    stock_code, start_date, end_date, period
                )

                if data_available:
                    self.download_status[stock_code]["completed"] = True
                    self.download_status[stock_code]["data_available"] = True
                    completed_stocks.append(stock_code)
                    current_completed.append(stock_code)
                    print(f"✅ {stock_code}: 数据验证可用")

            # 显示进度
            completed_count = len(completed_stocks)
            total_count = len(stock_codes)
            progress = (completed_count / total_count) * 100

            print(f"进度: {completed_count}/{total_count} ({progress:.1f}%)")

            if current_completed:
                print(f"本轮完成: {current_completed}")

            # 检查是否全部完成
            if completed_count == total_count:
                print("🎉 所有数据下载并验证完成!")
                return True, completed_stocks

            # 等待下一轮检查
            time.sleep(self.check_interval)

        # 超时处理
        print(f"⏰ 监控超时，已完成 {len(completed_stocks)}/{len(stock_codes)}")
        return False, completed_stocks

    def _check_data_available(
        self, stock_code: str, start_date: str, end_date: str, period: str
    ) -> bool:
        """检查数据是否可用"""
        try:
            data = xtdata.get_market_data(
                stock_list=[stock_code],
                period=period,
                start_time=start_date,
                end_time=end_date,
                count=10,  # 检查前10条数据
            )

            if stock_code in data and not data[stock_code].empty:
                df = data[stock_code]
                # 进一步验证数据质量
                if len(df) > 0 and "close" in df.columns:
                    return True
            return False

        except Exception:
            return False

    def get_download_status(self) -> Dict:
        """获取下载状态"""
        return self.download_status

    def load_data_after_download(
        self, stock_codes: List[str], start_date: str, end_date: str, period: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        在确保下载完成后加载数据
        """

        # 1. 确保下载完成
        all_completed, completed_stocks = self.download_with_guarantee(
            stock_codes, start_date, end_date, period
        )

        # 2. 加载数据
        stock_data_dict = {}

        if all_completed:
            load_codes = stock_codes
            print("加载所有股票数据...")
        else:
            load_codes = completed_stocks
            print(f"加载已完成的 {len(completed_stocks)} 只股票数据...")

        for stock_code in load_codes:
            try:
                data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period=period,
                    start_time=start_date,
                    end_time=end_date,
                )

                if stock_code in data and not data[stock_code].empty:
                    df = data[stock_code]
                    # 重置索引，确保日期列为普通列
                    if hasattr(df, "reset_index"):
                        df = df.reset_index()
                        if "index" in df.columns:
                            df = df.rename(columns={"index": "date"})

                    stock_data_dict[stock_code] = df
                    print(f"✅ {stock_code}: 加载成功 ({len(df)} 行)")
                else:
                    print(f"⚠️ {stock_code}: 数据加载为空")

            except Exception as e:
                print(f"❌ {stock_code}: 加载异常 - {e}")

        return stock_data_dict
