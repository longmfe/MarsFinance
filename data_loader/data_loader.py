import pandas as pd
from typing import Dict, Optional

class DataLoader:
    """数据加载器 - 基础实现"""
    
    def __init__(self):
        self.data_sources = {}
        
    def load_hs300_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        加载沪深300数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            股票数据字典
        """
        # 获取股票列表
        stock_codes = get_hs300_stock_list()
        print(f"获取到 {len(stock_codes)} 只股票")
        
        # 生成模拟数据
        print("生成股票数据...")
        stock_data_dict = generate_hs300_sample_data(
            stock_codes, 
            start_date=start_date 
            end_date=end_date
            )
     
        print(f"加载沪深300数据: {start_date} 到 {end_date}")
        return stock_data_dict 

    def generate_hs300_sample_data(stock_codes, start_date='20200101', end_date='20231231'):
        """生成沪深300股票的模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        stock_data_dict = {}
        fields=['open', 'close', 'high', 'low', 'volume', 'amount', 'preClose']
        xtdata.download_history_data2(stock_codes, '1d', start_date, end_date)

        for i, code in enumerate(stock_codes):
            print(f"\n正在测试第 {i+1} 只股票: {code}")
            data = xtdata.get_market_data_ex(field_list=fields, stock_list=[code], 
                                             start_time=start_date, end_time=end_date, period='1d', 
                                             count=1000)

            if data and code in data:
                df = data[code]
                stock_data_dict[code] = df
                print(f"成功! 数据形状: {df.shape}, 列名: {df.columns.tolist()}")
            else:
                print("获取失败或数据格式异常") 


        return stock_data_dict

    def get_hs300_stock_list():
        """
        获取沪深300成分股列表
    
        Returns:
            list: 包含沪深300成分股代码的列表，如果获取失败则返回空列表
        """
        try:
            # 获取沪深300成分股列表
            hs300_constituents = xtdata.get_stock_list_in_sector('沪深300')
            print(f"成功获取 {len(hs300_constituents)} 只沪深300成分股")
            print("前5只成分股示例:", hs300_constituents[:5])
            return hs300_constituents
        except Exception as e:
            print(f"获取沪深300成分股列表出错: {e}")
            return []  # 返回空列表而不是None，避免后续处理出错
    
    def add_data_source(self, name: str, source):
        """添加数据源"""
        self.data_sources[name] = source
        
    def list_available_data(self) -> list:
        """列出可用数据"""
        return list(self.data_sources.keys())

