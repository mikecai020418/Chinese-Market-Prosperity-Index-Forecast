import pmdarima as pm
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 设置中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

final_df = pd.read_csv('final_df.csv')

# 定义文件路径和当前日期限制
file_path = '收入数据_20241022_165734.xlsx'
current_date_limit = pd.Timestamp.today()

def load_and_prepare_df1(file_path):
    """
    加载 Excel 文件，提取频率行并清洗数据。
    
    参数:
        file_path (str): Excel 文件路径。
    
    返回:
        tuple: 一个元组，包含清洗后的数据框和列频率映射。
    """
    df = pd.read_excel(file_path)
    
    # 提取频率映射
    frequency_row = df.iloc[0].tolist()
    column_frequency = {
        df.columns[i]: frequency_row[i] 
        for i in range(1, len(df.columns)) if pd.notna(frequency_row[i])
    }
    
    # 删除频率行并重置数据框
    df = df.drop([0]).reset_index(drop=True)
    
    # 将 'Date' 列转换为日期类型
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    
    # 清洗数值列
    df = df.apply(clean_column)
    
    return df, column_frequency

def clean_column(col):
    """
    清洗列数据，将其转换为数值类型。
    
    参数:
        col (pandas.Series): 需要清洗的列。
    
    返回:
        pandas.Series: 清洗后的数值列。
    """
    if col.dtype == 'object':
        return pd.to_numeric(col.astype(str).str.replace(',', ''), errors='coerce')
    return pd.to_numeric(col, errors='coerce')

def shift_data_by_frequency(df, col, freq_type):
    """
    根据频率类型偏移数据。
    
    参数:
        df (pandas.DataFrame): 数据框。
        col (str): 需要偏移的列名。
        freq_type (str): 数据频率类型 ('日', '月', '季')。
    """
    if freq_type == '日':
        shifted_data = df[col].shift(-1)
    elif freq_type == '月':
        shifted_data = shift_monthly(df, col)
    elif freq_type == '季':
        shifted_data = shift_quarterly(df, col)
    df[col] = shifted_data

def shift_monthly(df, col):
    """
    偏移月度数据，将其移至下个月的10日。
    
    参数:
        df (pandas.DataFrame): 数据框。
        col (str): 需要偏移的列名。
    
    返回:
        pandas.Series: 偏移后的列数据。
    """
    shifted_data = df[col].shift(-8)
    shifted_data[(df.index + pd.DateOffset(months=1) + pd.offsets.Day(7)) > current_date_limit] = np.nan
    return shifted_data

def shift_quarterly(df, col):
    """
    偏移季度数据，将其移至下个月的20日。
    
    参数:
        df (pandas.DataFrame): 数据框。
        col (str): 需要偏移的列名。
    
    返回:
        pandas.Series: 偏移后的列数据。
    """
    shifted_data = df[col].shift(-15)
    shifted_data[(df.index + pd.DateOffset(months=1) + pd.offsets.Day(14)) > current_date_limit] = np.nan
    return shifted_data

def load_and_prepare_df2(df2):
    """
    加载 df2 数据，将 'date' 列转换为日期类型，并设置为索引。
    
    参数:
        df2 (pandas.DataFrame): 数据框。
    
    返回:
        pandas.DataFrame: 转换后的数据框。
    """
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)
    return df2

def align_date_ranges(df1, df2, start_date, end_date):
    """
    对齐 df1 和 df2 的日期范围，截取指定范围内的数据，并确保索引有序。
    
    参数:
        df1 (pandas.DataFrame): 第一个数据框。
        df2 (pandas.DataFrame): 第二个数据框。
        start_date (datetime): 起始日期。
        end_date (datetime): 结束日期。
    
    返回:
        tuple: 包含两个日期对齐后的数据框的元组。
    """
    # 确保索引有序
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    # 截取指定日期范围
    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    # 检查初步截取后的公共日期范围
    common_start = max(df1.index.min(), df2.index.min())
    common_end = min(df1.index.max(), df2.index.max())

    # 应用公共日期范围
    df1 = df1.loc[common_start:common_end]
    df2 = df2.loc[common_start:common_end]

    return df1, df2

def process_quarterly_and_accumulated_data(df, column, df2):
    """
    处理季度频率数据和“累计同比”列数据。

    参数:
    - df: 原始 DataFrame
    - column: 要处理的列名
    - df2: 用于对齐的 DataFrame

    返回:
    - 处理后匹配季度频率的 DataFrame
    """
    df1_clean = df[[column]].dropna()
    earliest_valid_date = df1_clean.index.min()
    df2_truncated = df2[df2.index >= earliest_valid_date]
    df1_quarterly = df1_clean.reindex(df2_truncated.index, method='ffill')
    return df1_quarterly

def direction_accuracy(actual, fitted):
    """
    计算方向准确率的函数

    参数:
    ----------
    actual : array_like
        实际值序列
    fitted : array_like
        拟合值序列

    返回值:
    ----------
    accuracy : float
        方向准确率，百分比形式
    """
    # 第一步：计算方向变化（即一阶差分）
    actual_diff = np.diff(actual, axis=0)
    fitted_diff = np.diff(fitted, axis=0)
    
    # 第二步：判断方向一致性（符号比较）
    direction_match = (np.sign(actual_diff) == np.sign(fitted_diff))
    
    # 第三步：计算方向准确率
    accuracy = np.mean(direction_match) * 100  # 转换为百分比
    
    return accuracy

# ARIMA 预测函数
def arima_forecast(df_column):
    """
    使用 ARIMA 模型对时间序列数据进行预测。

    参数:
        df_column (pd.Series): 用于预测的时间序列数据。

    返回:
        tuple: 包含预测值和置信区间的元组。
    """
    model = pm.auto_arima(df_column, seasonal=False, trace=True)
    forecast, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return forecast[0], conf_int


# ETS 预测函数
def ets_forecast(df_column):
    """
    使用指数平滑模型 (ETS) 对时间序列数据进行预测。

    参数:
        df_column (pd.Series): 用于预测的时间序列数据。

    返回:
        float: 预测值。
    """
    model = ExponentialSmoothing(df_column, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(1)
    return forecast[0]


# STL-ETS 预测函数
def stl_ets_forecast(df_column):
    """
    使用 STL 分解和 ETS 模型对时间序列数据进行预测。

    参数:
        df_column (pd.Series): 用于预测的时间序列数据。

    返回:
        float: 预测值。
    """
    stl = STL(df_column, seasonal=13)
    result = stl.fit()
    model = ExponentialSmoothing(result.trend.dropna(), trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(1)
    return forecast[0]


# Prophet 预测函数
def prophet_forecast(df_column):
    """
    使用 Prophet 模型对时间序列数据进行预测。

    参数:
        df_column (pd.Series): 用于预测的时间序列数据。

    返回:
        float: 预测值。
    """
    df_prophet = pd.DataFrame({'ds': df.index, 'y': df_column})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=1, freq='Q')
    forecast = model.predict(future)
    forecast_value = forecast.iloc[-1]['yhat']
    return forecast_value


# Holt-Winters 预测函数（三级指数平滑）
def holt_winters_forecast(df_column):
    """
    使用 Holt-Winters 模型对时间序列数据进行预测。

    参数:
        df_column (pd.Series): 用于预测的时间序列数据。

    返回:
        float: 预测值。
    """
    model = ExponentialSmoothing(df_column, trend='add', seasonal='add', seasonal_periods=4)
    model_fit = model.fit()
    forecast = model_fit.forecast(1)
    return forecast[0]

# 定义函数：基于 OLS 回归计算预测的 y 值
def calculate_forecast_y(forecast_avg):
    """
    使用 OLS 回归模型计算预测的 y 值，并生成置信区间。

    参数:
        forecast_avg (list): 每个主成分预测值的平均值组成的列表。

    返回:
        tuple: 包括预测的 y 值以及上下置信区间。
    """
    # 第一步：将 forecast_avg 转换为 NumPy 数组并调整形状用于预测
    forecast_avg = np.array(forecast_avg).reshape(1, -1)
    
    # 第二步：使用模型计算缩放后的预测 y 值
    forecast_y_scaled = model_ols.predict(forecast_avg)[0] 

    # 第三步：使用 get_prediction() 方法计算置信区间
    prediction_results = model_ols.get_prediction(forecast_avg)
    conf_int_scaled = prediction_results.conf_int()  # 获取置信区间的上下界

    # 第四步：提取置信区间的上下界
    conf_interval_low_scaled, conf_interval_high_scaled = conf_int_scaled[0]
    
    # 逆转换到原始 y 值
    forecast_y = scaler_y.inverse_transform([[forecast_y_scaled]])[0][0]
    conf_interval_low = scaler_y.inverse_transform([[conf_interval_low_scaled]])[0][0]
    conf_interval_high = scaler_y.inverse_transform([[conf_interval_high_scaled]])[0][0]
    
    return forecast_y, conf_interval_low, conf_interval_high


if __name__ == "__main__":
    # 加载并准备 df1 和 df2 数据
    df1, column_frequency = load_and_prepare_df1(file_path)
    df2 = load_and_prepare_df2(final_df.copy())

    # 根据列频率对数据进行时间偏移
    for col, freq in column_frequency.items():
        shift_data_by_frequency(df1, col, freq)

    # 对齐 df1 和 df2 的日期范围

    # 2006-06-30 - 64.29% acuuracy
    #使用：2006-12-31 - 65.22% accuracy
    # 2007-12-31 - 66.15%
    # 2008-03-31 - 67.19%
    # '2010-06-30' - 67.26%
    # '2010-09-30' - 68.52%
    # '2018-12-31' - 76.19%
    start_date = '2006-06-30'
    end_date = '2024-06-30'
    df1, df2 = align_date_ranges(df1, df2, start_date, end_date)
    # 创建空的 DataFrame 用于存储季度数据
    df1_quarterly_all = pd.DataFrame()

    # 处理季度数据和“累计同比”列
    for col in df1.columns:
        if column_frequency[col] == '季' or "累计同比" in col:
            df1_quarterly = process_quarterly_and_accumulated_data(df1, col, df2)
            df1_quarterly_all = pd.concat([df1_quarterly_all, df1_quarterly], axis=1)

    # 处理月度和日度数据以匹配季度索引
    for col, freq in column_frequency.items():
        if freq == '月' and "累计同比" not in col:
            # 将月度数据重采样为季度数据
            df1_quarterly = df1[col].resample('M').mean().resample('Q').mean()
            df1_quarterly_all[col] = df1_quarterly.reindex(df2.index, method='ffill')
        elif freq == '日' and "累计同比" not in col:
            # 将日度数据重采样为季度数据
            df1_quarterly = df1[col].resample('D').mean().resample('Q').mean()
            df1_quarterly_all[col] = df1_quarterly.reindex(df2.index, method='ffill')

    # 根据指定日期过滤最终 DataFrame
    df1_quarterly_all = df1_quarterly_all.loc[start_date:end_date]
    df1_quarterly_all = df1_quarterly_all.dropna()
    df1 = df1_quarterly_all
    globals()['df1'] = df1

    # 最后决定使用的因子
    x1_columns = [
        '交流电动机:产量:当月同比',  # 交流电动机
        '水泥专用设备:产量:当月同比',  # 水泥专用设备
        '全社会用电量:累计同比'      # 全社会用电量
    ]
    x2_columns = [
        '规模以上工业企业:亏损企业亏损总额:累计同比',  # 工业增加值
        '规模以上工业企业:利润总额:累计同比'       # 工业企业利润
    ]

    # 从 df1 中提取 x1 和 x2 数据，并对缺失值进行前向填充
    x1 = df1[x1_columns].fillna(method='ffill')
    x2 = df1[x2_columns].fillna(method='ffill')

    # 标准化 x1 数据
    scaler_x1 = StandardScaler()
    X1_scaled = scaler_x1.fit_transform(x1)
    # 标准化 x2 数据
    scaler_x2 = StandardScaler()
    X2_scaled = scaler_x2.fit_transform(x2)

    # 标准化目标变量 y（df2 数据）
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(np.array(df2.reindex(df1.index, method='ffill')).reshape(-1, 1))

    # 对 x1 进行 PCA
    n_components_x1 = min(X1_scaled.shape[0], X1_scaled.shape[1])
    pca_x1 = PCA(n_components=n_components_x1)
    pca_x1.fit(X1_scaled)

    # 获取解释方差比率并计算累计方差
    explained_variance_x1 = pca_x1.explained_variance_ratio_
    cumulative_variance_x1 = np.cumsum(explained_variance_x1)
    d_x1 = np.argmax(cumulative_variance_x1 > 0.9) + 1  # 解释>90%方差的成分数量
    print(f'Number of components for x1: {d_x1}')

    # 使用最优成分数重新拟合 x1 的 PCA
    pca_x1_opt = PCA(n_components=d_x1)
    pca_x1_components = pca_x1_opt.fit_transform(X1_scaled)

    # 对 x2 进行类似处理
    n_components_x2 = min(X2_scaled.shape[0], X2_scaled.shape[1])
    pca_x2 = PCA(n_components=n_components_x2)
    pca_x2.fit(X2_scaled)
    explained_variance_x2 = pca_x2.explained_variance_ratio_
    cumulative_variance_x2 = np.cumsum(explained_variance_x2)
    d_x2 = np.argmax(cumulative_variance_x2 > 0.9) + 1
    print(f'Number of components for x2: {d_x2}')
    pca_x2_opt = PCA(n_components=d_x2)
    pca_x2_components = pca_x2_opt.fit_transform(X2_scaled)

    # 将 x1 的 PCA 成分和 x2 的 PCA 成分合并
    pca_df = pd.DataFrame(pca_x1_components, columns=[f'X1_PC{i+1}' for i in range(d_x1)])
    pca_df = pd.concat([pca_df, pd.DataFrame(pca_x2_components, columns=[f'X2_PC{i+1}' for i in range(d_x2)])], axis=1)
    pca_df.index = df1.index  # 保持原始索引（日期）
    globals()['pca_df'] = pca_df
    # 使用 x1 和 x2 的 PCA 成分作为预测变量，y 作为目标变量进行回归
    # 第一步：识别 df1_quarterly_all 中不属于 x1_columns 或 x2_columns 的列
    other_columns = [col for col in df1_quarterly_all.columns if col not in x1_columns + x2_columns]

    # 第二步：过滤出 df1 时间范围内没有 NaN 值的列
    valid_other_columns = []
    for col in other_columns:
        # 检查该列在 df1 时间范围内是否包含 NaN 值
        if df1[col].loc[df1.index].isna().sum() == 0:
            valid_other_columns.append(col)
        else:
            print(f"列 '{col}' 在 df1 时间范围内包含 NaN 值，将被移除。")

    # 第三步：从 df1_quarterly_all 中提取有效的附加列
    other_data = df1_quarterly_all[valid_other_columns].loc[df1.index]
    globals()['other_data'] = other_data
    # 第四步：将 pca_components_combined 与附加列连接
    pca_components_combined = np.concatenate([pca_x1_components, pca_x2_components, other_data], axis=1)
    globals()['pca_components_combined'] = pca_components_combined
    # 将合并的 PCA 成分转换为 DataFrame，并使用指定索引
    model_ols = sm.OLS(y_scaled, pca_components_combined).fit()

    # 第一步：将缩放后的 y 反向变换回原始 y
    y_true = scaler_y.inverse_transform(y_scaled)  # scaler_y 是用于缩放 y 的 StandardScaler

    # 第二步：从模型中提取拟合值
    fitted_values_scaled = model_ols.fittedvalues  # 缩放形式的拟合值
    fitted_values = scaler_y.inverse_transform(fitted_values_scaled.reshape(-1, 1))  # 缩放回原始 y

    # 第四步：绘制拟合值与实际值的对比图，并设置自定义的 x 轴和 y 轴标签
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, y_true, label='实际值', color='blue')
    plt.plot(df1.index, fitted_values, label='拟合值', color='red', linestyle='--')
    plt.title('拟合值与实际值对比')
    plt.xlabel('日期')
    plt.ylabel('上证指数归母净利润同比（%)')
    plt.legend()

    # 第五步：将 x 轴主要刻度设置为 8 个月间隔
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))  # 设置主要刻度为每 8 个月
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化 x 轴刻度显示年月

    # 第六步：旋转 x 轴标签并调整布局
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示图表
    plt.show()

    # 应用到模型生成的 y_true 和拟合值
    y_true_values = y_true  # 确保是一维数组
    fitted_values_array = fitted_values

    accuracy_percentage = direction_accuracy(y_true_values, fitted_values_array)
    print(f"方向准确率: {accuracy_percentage:.2f}%")

    # 将所有列名组合成一个列表
    # 从两个数据框中获取列名并合并
    column_names = list(pca_df.columns) + list(other_data.columns)

    # 创建具有指定索引和列名的数据框
    df_pca_components_combined = pd.DataFrame(pca_components_combined, index=df1.index, columns=column_names)

    # 显示结果数据框
    df = df_pca_components_combined

    # 创建用于存储每列预测结果的字典
    forecast_results = {}

    # 遍历每一列并应用预测方法
    for col in df_pca_components_combined:
        print(f"正在预测 {col}")
        
        # ARIMA 预测
        forecast_arima, conf_arima = arima_forecast(df[col])
        
        # ETS 预测
        forecast_ets = ets_forecast(df[col])
        
        # STL-ETS 预测
        forecast_stl_ets = stl_ets_forecast(df[col])
        
        # Prophet 预测
        forecast_prophet = prophet_forecast(df[col])
        
        # Holt-Winters 预测
        forecast_holt_winters = holt_winters_forecast(df[col])
        
        # 存储结果
        forecast_results[col] = {
            'ARIMA': forecast_arima,
            'ETS': forecast_ets,
            'STL-ETS': forecast_stl_ets,
            'Prophet': forecast_prophet,
            'Holt-Winters': forecast_holt_winters
        }

        # 为每个模型绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[col], label='实际值')
        plt.axvline(pd.Timestamp('2024-06-30'), color='gray', linestyle='--')  # 标记当前日期
        
        # 绘制预测值
        plt.scatter(pd.Timestamp('2024-09-30'), forecast_arima, color='red', label='预测 (ARIMA)')
        plt.scatter(pd.Timestamp('2024-09-30'), forecast_ets, color='green', label='预测 (ETS)')
        plt.scatter(pd.Timestamp('2024-09-30'), forecast_stl_ets, color='purple', label='预测 (STL-ETS)')
        plt.scatter(pd.Timestamp('2024-09-30'), forecast_prophet, color='orange', label='预测 (Prophet)')
        plt.scatter(pd.Timestamp('2024-09-30'), forecast_holt_winters, color='blue', label='预测 (Holt-Winters)')
        
        plt.title(f'{col} 的预测结果')
        plt.legend()
        plt.show()

    # 显示预测结果
    for col, results in forecast_results.items():
        print(f"{col} 的预测结果:")
        for model_name, forecast in results.items():
            print(f"{model_name}: {forecast}")
        print("")

    # 第一步：将缩放后的 y 值逆转换回原始 y 值
    # 假设 scaler_y 是用于缩放 y 的 StandardScaler
    y_true = scaler_y.inverse_transform(y_scaled)

    # 第二步：从模型中提取拟合值并逆转换
    fitted_values_scaled = model_ols.fittedvalues  # 缩放形式的拟合值
    fitted_values = scaler_y.inverse_transform(fitted_values_scaled.reshape(-1, 1))  # 转换为原始 y 值
    # 第三步：计算每个主成分中各模型预测结果的平均值
    forecast_avg = []
    for col in df_pca_components_combined:
        avg_forecast = np.mean([
            forecast_results[col]['ARIMA'],
            forecast_results[col]['ETS'],
            forecast_results[col]['STL-ETS'],
            forecast_results[col]['Prophet'],
            forecast_results[col]['Holt-Winters']
        ])
        forecast_avg.append(avg_forecast)

    # 第四步：使用 OLS 模型计算预测的 y 值
    forecast_y, conf_int_low, conf_int_high = calculate_forecast_y(forecast_avg)

    # 第五步：绘制历史趋势与预测值及置信区间
    historical_dates = df.index
    historical_y_true = y_true  # 逆转换后的原始 y 值

    # 添加预测日期（2024-09-30）
    forecast_date = pd.Timestamp('2024-09-30')

    # 获取最后一个历史数据点
    last_hist_date = historical_dates[-1]  # 历史数据的最后日期
    last_hist_value = historical_y_true[-1]  # 历史数据的最后一个 y 值

    # 确保最后的历史值和预测值是标量
    last_hist_value = float(last_hist_value)  # 转换为标量
    forecast_y = float(forecast_y)  # 转换为标量

    # 绘制历史 y 值（原始、未缩放）
    plt.figure(figsize=(10, 6))
    plt.plot(historical_dates, historical_y_true, label='实际值 y (真实值)', color='blue')

    # 绘制 OLS 模型的拟合值
    plt.plot(historical_dates, fitted_values, label='拟合值 y (OLS)', color='green', linestyle='--')

    # 创建连续的置信区间
    plt.fill_between(
        [last_hist_date, forecast_date],
        [last_hist_value, conf_int_low],
        [last_hist_value, conf_int_high],
        color='red', alpha=0.3, label='置信区间'
    )

    # 绘制预测值
    plt.scatter(forecast_date, forecast_y, color='red', label='预测值 y', zorder=5)

    # 将最后的历史点与预测点连接
    plt.plot([last_hist_date, forecast_date], [last_hist_value, forecast_y], color='blue', linestyle='--', label='预测趋势')

    # 用竖线标记预测日期
    plt.axvline(forecast_date, color='gray', linestyle='--', label='预测日期')

    # 设置标题和轴标签
    plt.title('历史与预测 y 值（包含置信区间）')
    plt.xlabel('日期')
    plt.ylabel('上证指数归母净利润同比（%）')

    # 格式化 x 轴以显示日期
    plt.gcf().autofmt_xdate()

    # 显示图例
    plt.legend()

    # 显示绘图
    plt.show()

    # 输出预测结果
    print(f"2024-09-30 的预测 y 值: {forecast_y}")
    print(f"2024-06-30 的最后一个历史值: {last_hist_value}")
    print(f"95% 置信区间: [{conf_int_low}, {conf_int_high}]")



