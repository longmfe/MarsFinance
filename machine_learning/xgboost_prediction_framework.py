# 1. 环境准备与数据获取

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 技术指标计算库
import ta

# 机器学习相关
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_stock_data(ticker, start_date, end_date):
    """
    获取股票历史数据
    """
    print(f"正在下载 {ticker} 数据 ({start_date} 到 {end_date})...")
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock.empty:
        raise ValueError(f"无法获取 {ticker} 的数据")
    
    # 重命名列以便处理
    stock.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    
    print(f"获取到 {len(stock)} 个交易日数据")
    return stock

# 2. 特征工程：构建交易量与动量特征
def create_features(df):
    """
    创建预测特征，包括交易量特征和动量特征
    """
    df = df.copy()

    # ========== 基础价格特征 ==========
    df['Returns_1d'] = df['Close'].pct_change(1)  # 1日收益率
    df['Returns_5d'] = df['Close'].pct_change(5)  # 5日动量（核心特征）
    df['Returns_10d'] = df['Close'].pct_change(10)
    df['Returns_20d'] = df['Close'].pct_change(20)

    # 价格位置特征
    df['High_52w'] = df['Close'].rolling(window=252).max()
    df['Low_52w'] = df['Close'].rolling(window=252).min()
    df['Price_Position'] = (df['Close'] - df['Low_52w']) / (df['High_52w'] - df['Low_52w'])

    # ========== 交易量特征（重点） ==========

    # 1. 基础成交量特征
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio_5'] = df['Volume'] / df['Volume_MA_5']  # 成交量比率
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume_MA_20']

    # 2. 资金流强度指标 (Money Flow Intensity)
    # 典型价格
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    # 原始资金流
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']

    # 正向资金流（上涨日）
    df['Positive_MF'] = np.where(df['Close'] > df['Close'].shift(1),
                                  df['Raw_Money_Flow'], 0)
    # 负向资金流（下跌日）
    df['Negative_MF'] = np.where(df['Close'] < df['Close'].shift(1),
                                  df['Raw_Money_Flow'], 0)

    # 计算14日资金流比率（类似MFI但简化）
    df['Positive_MF_14'] = df['Positive_MF'].rolling(window=14).sum()
    df['Negative_MF_14'] = df['Negative_MF'].rolling(window=14).sum()
    df['MF_Ratio'] = df['Positive_MF_14'] / df['Negative_MF_14']
    df['Money_Flow_Index'] = 100 - (100 / (1 + df['MF_Ratio']))

    # 3. 量价背离特征
    # 价格创新高但成交量未创新高（顶背离）
    df['Price_New_High'] = df['Close'] == df['Close'].rolling(window=20).max()
    df['Volume_New_High'] = df['Volume'] == df['Volume'].rolling(window=20).max()
    df['Volume_Price_Divergence'] = np.where(
        (df['Price_New_High'] == True) & (df['Volume_New_High'] == False), 1, 0)

    # 4. 成交量波动率
    df['Volume_Volatility'] = df['Volume'].rolling(window=20).std() / df['Volume_MA_20']

    # 5. 价量相关性（过去20日）
    df['Price_Volume_Corr'] = df['Close'].rolling(window=20).corr(df['Volume'])

    # ========== 技术指标特征 ==========

    # 移动平均线
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_60'] = df['Close'].rolling(window=60).mean()

    # 价格与均线位置
    df['Price_vs_MA5'] = (df['Close'] - df['MA_5']) / df['MA_5']
    df['Price_vs_MA20'] = (df['Close'] - df['MA_20']) / df['MA_20']
    df['MA5_vs_MA20'] = (df['MA_5'] - df['MA_20']) / df['MA_20']

    # RSI (相对强弱指数)
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

    # 布林带
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # ========== 波动率特征 ==========
    df['Volatility_5d'] = df['Returns_1d'].rolling(window=5).std() * np.sqrt(252)
    df['Volatility_20d'] = df['Returns_1d'].rolling(window=20).std() * np.sqrt(252)

    # ========== 市场情绪特征 ==========
    # 连续上涨/下跌天数
    df['Up_Days'] = (df['Close'] > df['Close'].shift(1)).rolling(window=5).sum()
    df['Down_Days'] = (df['Close'] < df['Close'].shift(1)).rolling(window=5).sum()

    # 价格加速度（动量的动量）
    df['Momentum_Acceleration'] = df['Returns_5d'] - df['Returns_5d'].shift(5)

    # ========== 目标变量：未来30天收益率 ==========
    df['Target_30d'] = df['Close'].pct_change(30).shift(-30)

    # 删除包含NaN的行（由于滚动计算）
    df_clean = df.dropna()

    print(f"特征工程完成，创建了 {df_clean.shape[1] - 1} 个特征")
    print(f"有效数据样本数: {df_clean.shape[0]}")

    return df_clean

# 3. 数据预处理与特征选择
def prepare_data(df, test_size=0.2):
    """
    准备训练和测试数据
    """
    # 分离特征和目标
    feature_columns = [col for col in df.columns if col not in ['Target_30d', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']]
    X = df[feature_columns]
    y = df['Target_30d']

    # 按时间顺序划分训练集和测试集（时间序列数据重要！）
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征数量: {X_train.shape[1]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

# 4. XGBoost模型构建与训练
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost回归模型
    """
    # 定义XGBoost参数
    params = {
        'objective': 'reg:squarederror',  # 回归任务
        'n_estimators': 300,              # 树的数量
        'learning_rate': 0.05,            # 学习率
        'max_depth': 6,                   # 树的最大深度
        'min_child_weight': 1,            # 最小叶子节点样本权重和
        'subsample': 0.8,                 # 样本采样比例
        'colsample_bytree': 0.8,          # 特征采样比例
        'gamma': 0,                       # 分裂所需的最小损失减少
        'reg_alpha': 0.1,                 # L1正则化
        'reg_lambda': 1,                  # L2正则化
        'random_state': 42,
        'n_jobs': -1                      # 使用所有CPU核心
    }

    # 创建模型
    model = xgb.XGBRegressor(**params)

    # 训练模型
    print("开始训练XGBoost模型...")
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric=['rmse', 'mae'],
              verbose=False,
              early_stopping_rounds=20)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return model, y_train_pred, y_test_pred

def evaluate_model(y_true, y_pred, dataset_name="测试集"):
    """
    评估模型性能
    """
    # 回归指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # 方向准确性（预测涨跌方向）
    direction_correct = np.sum((y_true > 0) == (y_pred > 0)) / len(y_true) * 100

    # 转换为百分比
    mse_pct = mse * 10000  # 转换为基点平方
    mae_pct = mae * 100    # 转换为百分比
    rmse_pct = rmse * 100  # 转换为百分比

    print(f"\n{'='*50}")
    print(f"{dataset_name} 评估结果:")
    print(f"{'='*50}")
    print(f"均方误差 (MSE): {mse_pct:.2f} 基点²")
    print(f"平均绝对误差 (MAE): {mae_pct:.2f}%")
    print(f"均方根误差 (RMSE): {rmse_pct:.2f}%")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"方向准确性: {direction_correct:.2f}%")

    return {
        'mse': mse, 'mae': mae, 'rmse': rmse,
        'r2': r2, 'direction_accuracy': direction_correct
    }

# 5. 特征重要性分析与可视化
def analyze_feature_importance(model, feature_names, top_n=20):
    """
    分析特征重要性
    """
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"\n{'='*50}")
    print("特征重要性排名 (前20名):")
    print('='*50)

    for i, row in importance_df.head(top_n).iterrows():
        print(f"{i+1:2d}. {row['feature']:30s} : {row['importance']:.4f}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'][::-1])
    plt.yticks(range(len(top_features)), top_features['feature'][::-1])
    plt.xlabel('特征重要性')
    plt.title('XGBoost特征重要性排名 (前20名)')
    plt.tight_layout()
    plt.show()

    # 按特征类别分组分析
    feature_categories = {
        '动量特征': ['Returns_5d', 'Returns_10d', 'Returns_20d', 'Momentum_Acceleration'],
        '交易量特征': ['Volume_Ratio_5', 'Volume_Ratio_20', 'Money_Flow_Index',
                    'Volume_Price_Divergence', 'Volume_Volatility', 'Price_Volume_Corr'],
        '技术指标': ['RSI_14', 'MACD_Diff', 'BB_Width', 'BB_Position',
                  'Price_vs_MA5', 'Price_vs_MA20', 'MA5_vs_MA20'],
        '波动率': ['Volatility_5d', 'Volatility_20d'],
        '市场情绪': ['Up_Days', 'Down_Days', 'Price_Position']
    }

    category_importance = {}
    for category, features in feature_categories.items():
        # 找出实际存在的特征
        existing_features = [f for f in features if f in importance_df['feature'].values]
        if existing_features:
            category_importance[category] = importance_df[
                importance_df['feature'].isin(existing_features)
            ]['importance'].sum()

    # 绘制类别重要性
    plt.figure(figsize=(10, 6))
    categories = list(category_importance.keys())
    importances = [category_importance[cat] for cat in categories]

    plt.pie(importances, labels=categories, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('特征类别重要性分布')
    plt.show()

    return importance_df, category_importance

# 6. 预测结果可视化
def visualize_predictions(y_true, y_pred, dates=None, title="预测结果对比"):
    """
    可视化预测结果
    """
    plt.figure(figsize=(14, 10))

    # 实际值 vs 预测值散点图
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('实际30天收益率')
    plt.ylabel('预测30天收益率')
    plt.title('实际值 vs 预测值')
    plt.grid(True, alpha=0.3)

    # 预测误差分布
    plt.subplot(2, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    plt.grid(True, alpha=0.3)

    # 时间序列对比（如果有日期）
    if dates is not None and len(dates) == len(y_true):
        plt.subplot(2, 1, 2)
        plt.plot(dates, y_true, 'b-', label='实际值', alpha=0.7, linewidth=1)
        plt.plot(dates, y_pred, 'r-', label='预测值', alpha=0.7, linewidth=1)
        plt.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')
        plt.xlabel('日期')
        plt.ylabel('30天收益率')
        plt.title('时间序列：实际值 vs 预测值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 方向准确性分析
    correct_up = np.sum((y_true > 0) & (y_pred > 0))
    correct_down = np.sum((y_true < 0) & (y_pred < 0))
    wrong_up = np.sum((y_true < 0) & (y_pred > 0))
    wrong_down = np.sum((y_true > 0) & (y_pred < 0))

    confusion_data = [[correct_up, wrong_up], [wrong_down, correct_down]]
    confusion_df = pd.DataFrame(confusion_data,
                                index=['实际上涨', '实际下跌'],
                                columns=['预测上涨', '预测下跌'])

    print("\n预测方向混淆矩阵:")
    print(confusion_df)

    # 计算精确率、召回率
    precision_up = correct_up / (correct_up + wrong_up) if (correct_up + wrong_up) > 0 else 0
    recall_up = correct_up / (correct_up + wrong_down) if (correct_up + wrong_down) > 0 else 0
    precision_down = correct_down / (correct_down + wrong_down) if (correct_down + wrong_down) > 0 else 0
    recall_down = correct_down / (correct_down + wrong_up) if (correct_down + wrong_up) > 0 else 0

    print(f"\n上涨预测精确率: {precision_up:.2%}")
    print(f"上涨预测召回率: {recall_up:.2%}")
    print(f"下跌预测精确率: {precision_down:.2%}")
    print(f"下跌预测召回率: {recall_down:.2%}")

# 7. 主程序：端到端预测流程
def main_pipeline(ticker="AAPL", years_of_data=5, test_size=0.2):
    """
    完整的端到端预测流程
    """
    print("="*60)
    print(f"股票未来30天收益率预测系统")
    print(f"标的: {ticker}")
    print("="*60)

    # 1. 获取数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_of_data*365)

    try:
        raw_data = fetch_stock_data(ticker, start_date, end_date)
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

    # 2. 特征工程
    print("\n1. 进行特征工程...")
    df_with_features = create_features(raw_data)

    # 3. 数据准备
    print("\n2. 准备训练和测试数据...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        df_with_features, test_size=test_size)

    # 4. 训练模型
    print("\n3. 训练XGBoost模型...")
    model, y_train_pred, y_test_pred = train_xgboost_model(
        X_train, y_train, X_test, y_test)

    # 5. 评估模型
    print("\n4. 评估模型性能...")
    train_metrics = evaluate_model(y_train, y_train_pred, "训练集")
    test_metrics = evaluate_model(y_test, y_test_pred, "测试集")

    # 6. 特征重要性分析
    print("\n5. 分析特征重要性...")
    importance_df, category_importance = analyze_feature_importance(
        model, feature_names)

    # 7. 可视化结果
    print("\n6. 可视化预测结果...")
    test_dates = df_with_features.index[-len(y_test):]
    visualize_predictions(y_test, y_test_pred, test_dates)

    # 8. 生成预测报告
    print("\n" + "="*60)
    print("预测报告摘要")
    print("="*60)

    # 找出最重要的特征
    top_features = importance_df.head(5)['feature'].tolist()

    print(f"\n🔍 关键发现:")
    print(f"1. 最重要的5个特征: {', '.join(top_features)}")

    # 检查5日动量特征的重要性
    momentum_rank = importance_df[importance_df['feature'] == 'Returns_5d'].index
    if not momentum_rank.empty:
        rank = momentum_rank[0] + 1
        print(f"2. '5日动量'特征重要性排名: 第{rank}位")

    # 检查资金流特征的重要性
    mfi_rank = importance_df[importance_df['feature'] == 'Money_Flow_Index'].index
    if not mfi_rank.empty:
        rank = mfi_rank[0] + 1
        print(f"3. '资金流指数'特征重要性排名: 第{rank}位")

    print(f"\n📈 模型表现:")
    print(f"测试集方向准确性: {test_metrics['direction_accuracy']:.1f}%")
    print(f"测试集R²分数: {test_metrics['r2']:.4f}")

    # 判断模型实用性
    if test_metrics['direction_accuracy'] > 55:
        print(f"\n✅ 模型显示出一定的预测能力")
        if test_metrics['direction_accuracy'] > 60:
            print("   预测能力较强，可考虑用于辅助决策")
    else:
        print(f"\n⚠️  模型预测能力有限，建议进一步优化")

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_metrics': test_metrics,
        'predictions': y_test_pred,
        'actuals': y_test
    }

# 8. 使用示例与模型部署
# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例1：分析苹果股票
    results = main_pipeline(ticker="AAPL", years_of_data=5, test_size=0.2)

    # 示例2：批量分析多只股票
    # tickers = ["MSFT", "GOOGL", "AMZN", "TSLA"]
    # all_results = {}
    # for ticker in tickers:
    #     print(f"\n{'='*60}")
    #     print(f"分析 {ticker}")
    #     print('='*60)
    #     try:
    #         all_results[ticker] = main_pipeline(ticker=ticker, years_of_data=3, test_size=0.2)
    #     except Exception as e:
    #         print(f"分析{ticker}时出错: {e}")

    # 示例3：使用训练好的模型进行单点预测
    def predict_single_point(model, scaler, recent_data, feature_names):
        """
        使用训练好的模型进行单点预测
        recent_data: 包含最近60个交易日数据的DataFrame
        """
        # 创建特征
        df_features = create_features(recent_data)

        if len(df_features) == 0:
            return None

        # 获取最新数据点
        latest_features = df_features.iloc[-1:][feature_names]

        # 标准化
        latest_scaled = scaler.transform(latest_features)

        # 预测
        prediction = model.predict(latest_scaled)[0]

        print(f"\n预测未来30天收益率: {prediction*100:.2f}%")
        if prediction > 0:
            print("预测方向: 上涨 📈")
        else:
            print("预测方向: 下跌 📉")

        return prediction
