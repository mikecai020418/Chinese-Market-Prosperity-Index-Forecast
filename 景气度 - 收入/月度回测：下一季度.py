import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from prophet import Prophet

# 设置中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

final_df = pd.read_csv('final_df.csv')
cmap = plt.get_cmap('tab10') 

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

# 定义函数以获取最近的有效值
def get_latest_valid_value(column, date):
    """
    从给定日期向后搜索，直到找到有效值。
    """
    # 从'日期'开始向后搜索，直到找到有效值
    while pd.isna(df1.loc[date, column]):
        previous_dates = df1.loc[:date].index  # 获取之前的日期索引
        if len(previous_dates) > 1:
            date = previous_dates[-2]  # 回退到前一个日期
        else:
            return np.nan  # 如果没有更早的日期，返回NaN
    return df1.loc[date, column]

# 定义辅助函数
def prepare_data(df, columns):
    """
    对指定列的数据进行标准化处理。
    
    参数:
    df (DataFrame): 包含数据的DataFrame。
    columns (list): 需要标准化的列名列表。
    
    返回:
    tuple: 标准化器对象和标准化后的数据。
    """
    subset = df[columns].ffill()  # 使用前向填充处理缺失值
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(subset)

def apply_pca(X_scaled, explained_variance_threshold=0.9):
    """
    对标准化后的数据应用PCA，并返回覆盖指定方差阈值的主成分。
    
    参数:
    X_scaled (ndarray): 标准化后的输入数据。
    explained_variance_threshold (float): 累积解释方差比例的阈值，默认为90%。
    
    返回:
    tuple: PCA模型、转换后的数据、主成分数量。
    """
    pca = PCA().fit(X_scaled)
    num_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > explained_variance_threshold) + 1
    pca_opt = PCA(n_components=num_components).fit(X_scaled)
    return pca_opt, pca_opt.transform(X_scaled), num_components

def fit_ols(y, X):
    """
    拟合普通最小二乘(OLS)模型。
    
    参数:
    y (ndarray): 目标变量。
    X (ndarray): 特征变量。
    
    返回:
    RegressionResults: OLS模型拟合结果。
    """
    model = sm.OLS(y, X).fit()
    return model

# 第 1 步：为每个 PCA 组件定义五种预测模型

def arima_forecast(df_column):
    """
    使用 ARIMA 模型进行预测。
    
    参数：
    df_column (pd.Series): 输入的时间序列数据
    
    返回：
    float: ARIMA 模型预测的下一个值
    """
    # df_column = df_column.copy()  # 如果需要的话，可以解除注释以避免修改原数据
    # df_column.index = pd.to_datetime(df_column.index).to_period('Q').to_timestamp()  # 将索引转换为季度时间戳
    model = pm.auto_arima(df_column, seasonal=False, trace=False)  # 使用自动 ARIMA 模型
    forecast, conf_int = model.predict(n_periods=1, return_conf_int=True)  # 预测下一个周期
    return forecast.iloc[0]  # 返回预测值

def ets_forecast(df_column):
    """
    使用指数平滑（Exponential Smoothing）模型进行预测。
    
    参数：
    df_column (pd.Series): 输入的时间序列数据
    
    返回：
    float: 指数平滑模型预测的下一个值
    """
    model = ExponentialSmoothing(df_column, trend='add', seasonal=None)  # 定义指数平滑模型，趋势项为加性
    model_fit = model.fit()  # 拟合模型
    forecast = model_fit.forecast(1)  # 预测下一个值
    return forecast.iloc[0]  # 返回预测值

def stl_ets_forecast(df_column):
    """
    使用 STL 分解和指数平滑（ETS）模型进行预测。
    
    参数：
    df_column (pd.Series): 输入的时间序列数据
    
    返回：
    float: 使用 STL 和 ETS 模型预测的下一个值
    
    异常：
    ValueError: 如果 STL 分解后的趋势组件为空，抛出异常
    """
    df_column1 = df_column.copy()  # 创建数据副本，避免修改原数据
    
    # 备份原始索引
    original_index = df_column1.index
    # 转换为季度频率以供 STL 使用
    df_column1 = df_column1.asfreq('Q')
    stl = STL(df_column1, seasonal=13)  # 使用 STL 进行季节性分解
    result = stl.fit()  # 拟合 STL 模型

    # 检查趋势组件是否为空
    trend = result.trend.dropna()
    if len(trend) == 0:
        raise ValueError("趋势组件为空，无法进行 ETS 预测。")  # 如果趋势组件为空，抛出异常
    
    # 在趋势组件上拟合 ETS 模型
    model = ExponentialSmoothing(trend, trend='add', seasonal=None)
    model_fit = model.fit()
    
    # 预测下一个值
    forecast = model_fit.forecast(1)
    # 恢复原始索引（如果以后需要使用原始索引）
    df_column.index = original_index
    
    return forecast.iloc[0]  # 返回预测值

def prophet_forecast(df_column):
    """
    使用 Prophet 模型进行预测。
    
    参数：
    df_column (pd.Series): 输入的时间序列数据
    
    返回：
    float: Prophet 模型预测的下一个值
    """
    # 将索引对齐以匹配数据列的长度
    df_prophet = pd.DataFrame({
        'ds': df.index[:len(df_column)],  # 使用数据的时间索引
        'y': df_column
    })
    
    model = Prophet()  # 创建 Prophet 模型
    model.fit(df_prophet)  # 拟合模型
    future = model.make_future_dataframe(periods=1, freq='Q')  # 生成未来一个季度的预测
    forecast = model.predict(future)  # 预测未来的值
    forecast_value = forecast.iloc[-1]['yhat']  # 获取最后一个预测值
    
    return forecast_value  # 返回预测值

def holt_winters_forecast(df_column):
    """
    使用 Holt-Winters 模型进行预测。
    
    参数：
    df_column (pd.Series): 输入的时间序列数据
    
    返回：
    float: Holt-Winters 模型预测的下一个值
    """
    model = ExponentialSmoothing(df_column, trend='add', seasonal='add', seasonal_periods=4)  # 定义 Holt-Winters 模型
    model_fit = model.fit()  # 拟合模型
    forecast = model_fit.forecast(1)  # 预测下一个值
    return forecast.iloc[0]  # 返回预测值

# 计算每个月预测的准确性
def calculate_accuracy(forecasts, actuals):
    """
    计算预测结果的平均绝对误差（MAE）和均方根误差（RMSE）。
    :param forecasts: 预测值
    :param actuals: 实际值
    :return: (MAE, RMSE)
    """
    mae = mean_absolute_error(actuals, forecasts)
    rmse = np.sqrt(mean_squared_error(actuals, forecasts))
    return mae, rmse


# 定义一个函数用于分类变化方向
def categorize_direction(change):
    """
    分类变化方向的函数
    如果变化值大于 0，返回 'Positive'（正增长）
    如果变化值小于 0，返回 'Negative'（负增长）
    如果变化值为 0，返回 'Even'（无变化）
    
    参数：
    change (float): 变化值
    
    返回：
    str: 分类结果，'Positive'，'Negative'，或 'Even'
    """
    if change > 0:
        return 'Positive'  # 正增长
    elif change < 0:
        return 'Negative'  # 负增长
    else:
        return 'Even'  # 无变化

# 比较实际值和预测值的方向以计算准确率
def calculate_direction_accuracy(actual, forecast):
    """
    计算方向准确率的函数
    比较实际和预测的方向是否一致，计算一致的比例
    
    参数：
    actual (pd.Series): 实际变化方向
    forecast (pd.Series): 预测变化方向
    
    返回：
    float: 方向准确率，以百分比表示
    """
    correct_directions = (actual == forecast)
    accuracy = correct_directions.sum() / len(correct_directions) * 100
    return accuracy

if __name__ == "__main__":
    # 加载并准备 df1 和 df2 数据
    df1, column_frequency = load_and_prepare_df1(file_path)
    df2 = load_and_prepare_df2(final_df.copy())

    # 根据列频率对数据进行时间偏移
    for col, freq in column_frequency.items():
        shift_data_by_frequency(df1, col, freq)

    # 定义日期参数
    training_end_date = '2020-09-30'  # 训练集结束日期
    testing_start_date = '2020-10-30'  # 测试集开始日期
    testing_end_date = '2024-06-30'  # 测试集结束日期

    # 确保DataFrame的索引是按时间排序的
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    # 确保索引为DatetimeIndex类型
    df1.index = pd.to_datetime(df1.index)

    # 生成完整的日期范围，从最小日期到最大日期
    full_date_range = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='D')

    # 重新索引DataFrame以包含所有日期
    df1 = df1.reindex(full_date_range)

    # 定义相关列名
    x1_columns = [
        '交流电动机:产量:当月同比',  # 示例变量1
        '水泥专用设备:产量:当月同比',  # 示例变量2
        '全社会用电量:累计同比'  # 示例变量3
    ]
    x2_columns = [
        '规模以上工业企业:亏损企业亏损总额:累计同比',  # 示例变量1
        '规模以上工业企业:利润总额:累计同比'  # 示例变量2
    ]


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

    start_date = '2006-06-30'
    end_date = '2024-06-30'
    # 根据指定日期过滤最终 DataFrame
    df1_quarterly_all = df1_quarterly_all.loc[start_date:end_date]
    df1_quarterly_all = df1_quarterly_all.dropna()

    # 准备季度数据子集
    df1_quarterly_training = df1_quarterly_all.loc['2006-12-31':training_end_date]  # 从2006-12-31到训练结束日期
    df2_quarterly_training = df2.loc['2006-12-31':training_end_date]  # 同期的目标变量

    # 准备并标准化训练数据
    scaler_x1, X1_train_scaled = prepare_data(df1_quarterly_training, x1_columns)
    scaler_x2, X2_train_scaled = prepare_data(df1_quarterly_training, x2_columns)
    scaler_y, y_train_scaled = prepare_data(df2_quarterly_training, df2.columns)

    # 对每个特征集合应用PCA
    pca_x1_opt, X1_train_pca, d_x1 = apply_pca(X1_train_scaled)
    pca_x2_opt, X2_train_pca, d_x2 = apply_pca(X2_train_scaled)

    # 合并PCA结果并生成新的DataFrame
    pca_df = pd.concat([
        pd.DataFrame(X1_train_pca, columns=[f'X1_PC{i+1}' for i in range(d_x1)], index=df1_quarterly_training.index),
        pd.DataFrame(X2_train_pca, columns=[f'X2_PC{i+1}' for i in range(d_x2)], index=df1_quarterly_training.index)
    ], axis=1)

    # 提取其他非PCA的变量列
    other_columns = [col for col in df1_quarterly_training.columns if col not in x1_columns + x2_columns]
    other_data = df1_quarterly_training[other_columns].dropna(axis=1)  # 去除含NaN值的列

    # 合并PCA转换后的主成分和其他数据
    X_train = pd.concat([pca_df, other_data], axis=1)

    # 拟合初始OLS模型
    model_ols = fit_ols(y_train_scaled, X_train)

    # 准备测试阶段的月份范围
    test_months = pd.date_range(start=testing_start_date, end=testing_end_date, freq='M')

    # 收集不同时间的实际值和拟合值
    fitted_y_across_time = []
    actual_y_across_time = []

    # 绘图准备：获取训练阶段的初始拟合值
    initial_fitted_y_scaled = model_ols.fittedvalues
    initial_fitted_y = scaler_y.inverse_transform(initial_fitted_y_scaled.to_numpy().reshape(-1, 1))
    full_actual_y = df2.loc['2006-12-31':testing_end_date].values

    # 绘制实际值与拟合值的对比图
    plt.figure(figsize=(12, 8))

    # 绘制实际值曲线（2006-12-31到2024-06-30）
    plt.plot(df2.loc['2006-12-31':testing_end_date].index, full_actual_y, linestyle='--', label='实际值', color='blue')

    # 定义三种测试月份预测
    month_1_forecasts = []  # 第一种预测
    month_2_forecasts = []  # 第二种预测
    month_3_forecasts = []  # 第三种预测
    month_1_dates = []  # 第一种预测的日期
    month_2_dates = []  # 第二种预测的日期
    month_3_dates = []  # 第三种预测的日期

    # Initialize lists to store forecast errors
    month_1_errors = []
    month_2_errors = []
    month_3_errors = []

    # 第 2 步：使用实际历史值实现逐步预测
    for i, test_month in enumerate(test_months):
        # 确定当前是季度的第几个月（第一、二或第三个月）
        month_in_quarter = (test_month.month - 1) % 3 + 1

        # 获取当前月份的数据，如果没有找到则使用最近的日期
        if test_month not in df1.index:
            test_month = df1.index[df1.index.get_loc(test_month, method='nearest')]

        # 初始化当前月份的 X1、X2 和其他特征的空 DataFrame
        current_x1 = pd.DataFrame(index=[test_month], columns=x1_columns)
        current_x2 = pd.DataFrame(index=[test_month], columns=x2_columns)
        current_other = pd.DataFrame(index=[test_month], columns=other_columns)

        # 针对每一列，追溯直到找到有效值
        for col in x1_columns:
            if pd.isna(df1.loc[test_month, col]):  # 如果当前月份的值是 NaN
                current_x1[col] = get_latest_valid_value(col, test_month)
            else:
                current_x1[col] = df1.loc[test_month, col]  # 如果有效，直接使用当前月份的值

        for col in x2_columns:
            if pd.isna(df1.loc[test_month, col]):  # 如果当前月份的值是 NaN
                current_x2[col] = get_latest_valid_value(col, test_month)
            else:
                current_x2[col] = df1.loc[test_month, col]  # 如果有效，直接使用当前月份的值

        for col in other_columns:
            if pd.isna(df1.loc[test_month, col]):  # 如果当前月份的值是 NaN
                current_other[col] = get_latest_valid_value(col, test_month)
            else:
                current_other[col] = df1.loc[test_month, col]  # 如果有效，直接使用当前月份的值

        # 计算季度结束月份
        quarter_end_month = test_month + pd.DateOffset(months=(3 - month_in_quarter)) + pd.offsets.MonthEnd(0)
        
        # 处理不同季度月份的情况
        if month_in_quarter == 1:
            # 如果是季度的第一个月：直接使用当前的值
            quarter_x1 = current_x1
            quarter_x2 = current_x2
            quarter_other = current_other
            actual_y_across_time.append(None)

            X1_test_scaled = scaler_x1.transform(quarter_x1)
            X2_test_scaled = scaler_x2.transform(quarter_x2)

            # 使用预先拟合的 PCA 组件进行转换
            pca_x1_test = pca_x1_opt.transform(X1_test_scaled)
            pca_x2_test = pca_x2_opt.transform(X2_test_scaled)

            # 合并测试数据的 PCA 组件
            try:
                pca_test_combined = np.concatenate([pca_x1_test, pca_x2_test, quarter_other], axis=1)
            except ValueError as e:
                print("合并错误:", e)
            
            # 确保包含 x1 和 x2
            pca_test_df = pd.DataFrame(pca_test_combined, columns=X_train.columns)
            pca_test_df.index = [quarter_end_month]
            
            # 使用 pd.concat 将 pca_test_df 追加到 pca_train_df
            pca_combined_df = pd.concat([X_train, pca_test_df])

            forecast_avg = []
            
            df = pca_combined_df

            for col in pca_combined_df.columns:
                # 对每一列数据，截取当前预测点之前的数据
                df_column_actual = pca_combined_df[col]
                # 确保索引类型正确（如果必要，重置为 datetime 索引）
                # 使用历史数据进行每个 PCA 组件的预测
                forecast_arima = arima_forecast(df_column_actual)
                forecast_ets = ets_forecast(df_column_actual)
                forecast_stl_ets = stl_ets_forecast(df_column_actual)
                forecast_prophet = prophet_forecast(df_column_actual)
                forecast_holt_winters = holt_winters_forecast(df_column_actual)
                
                # 对这些预测值取平均
                avg_forecast = np.mean([
                    forecast_arima,
                    forecast_ets,
                    forecast_stl_ets,
                    forecast_prophet,
                    forecast_holt_winters
                ])
                forecast_avg.append(avg_forecast)

            # 使用 OLS 模型预测 y
            forecast_y_scaled = model_ols.predict(forecast_avg)
            forecast_y = scaler_y.inverse_transform(forecast_y_scaled.reshape(-1, 1))
            month_1_forecasts.append(forecast_y[0][0])
            month_1_dates.append(test_month)
            fitted_y_across_time.append(forecast_y[0][0])
            plt.scatter(test_month, forecast_y[0][0], color='blue', label=f'预测 y（下季度，{test_month.strftime("%Y-%m")})')

        elif month_in_quarter == 2:
            # 第二个月：平均当前月份和上个月的值
            last_month = test_month - pd.DateOffset(months=1)

            # 初始化 last_x1 和 last_x2 为 DataFrame
            last_x1 = pd.DataFrame(index=[last_month], columns=x1_columns)
            last_x2 = pd.DataFrame(index=[last_month], columns=x2_columns)
            last_other = pd.DataFrame(index=[last_month], columns=other_columns)

            # 获取当前月份和上个月份的每一列的非 NaN 值
            for col in x1_columns:
                if pd.isna(df1.loc[test_month, col]):
                    current_x1[col] = get_latest_valid_value(col, test_month)  # 获取当前月份的有效值
                else:
                    current_x1[col] = df1.loc[test_month, col]  # 当前月份存在有效值，直接使用

                if pd.isna(df1.loc[last_month, col]):
                    last_x1[col] = get_latest_valid_value(col, last_month)  # 获取上个月份的有效值
                else:
                    last_x1[col] = df1.loc[last_month, col]  # 上个月份存在有效值，直接使用

            for col in x2_columns:
                if pd.isna(df1.loc[test_month, col]):
                    current_x2[col] = get_latest_valid_value(col, test_month)  # 获取当前月份的有效值
                else:
                    current_x2[col] = df1.loc[test_month, col]  # 当前月份存在有效值，直接使用

                if pd.isna(df1.loc[last_month, col]):
                    last_x2[col] = get_latest_valid_value(col, last_month)  # 获取上个月份的有效值
                else:
                    last_x2[col] = df1.loc[last_month, col]  # 上个月份存在有效值，直接使用

            for col in other_columns:
                if pd.isna(df1.loc[test_month, col]):  # 如果当前月份的值为 NaN
                    current_other[col] = get_latest_valid_value(col, test_month)  # 获取当前月份的有效值
                else:
                    current_other[col] = df1.loc[test_month, col]  # 当前月份存在有效值，直接使用

                if pd.isna(df1.loc[last_month, col]):
                    last_other[col] = get_latest_valid_value(col, last_month)  # 获取上个月份的有效值
                else:
                    last_other[col] = df1.loc[last_month, col]  # 上个月份存在有效值，直接使用

            # 平均当前月和上个月的值
            # 平均 current_x1 和 last_x1 的值，忽略索引
            quarter_x1_values = (current_x1.values + last_x1.values) / 2
            quarter_x1 = pd.DataFrame(quarter_x1_values, index=current_x1.index, columns=current_x1.columns)
            # 平均 current_x2 和 last_x2 的值，忽略索引
            quarter_x2_values = (current_x2.values + last_x2.values) / 2
            quarter_x2 = pd.DataFrame(quarter_x2_values, index=current_x2.index, columns=current_x2.columns)

            quarter_other_values = (current_other.values + last_other.values) / 2
            quarter_other = pd.DataFrame(quarter_other_values, index=current_other.index, columns=current_other.columns)
            
            # 标准化和变换数据
            X1_test_scaled = scaler_x1.transform(quarter_x1)
            X2_test_scaled = scaler_x2.transform(quarter_x2)

            # 使用预先拟合的 PCA 组件进行变换
            pca_x1_test = pca_x1_opt.transform(X1_test_scaled)
            pca_x2_test = pca_x2_opt.transform(X2_test_scaled)

            try:
                pca_test_combined = np.concatenate([pca_x1_test, pca_x2_test, quarter_other], axis=1)  # 合并 PCA 组件和其他特征
            except ValueError as e:
                print("Concatenation error:", e)

            pca_test_df = pd.DataFrame(pca_test_combined, columns=X_train.columns)
            pca_test_df.index = [quarter_end_month]
            
            # 使用 pd.concat 将 pca_test_df 和 pca_train_df 进行拼接
            pca_combined_df = pd.concat([X_train, pca_test_df])
            
            forecast_avg = []  # 用于存放所有预测结果的列表
            df = pca_combined_df

            for col in pca_combined_df.columns:
                # 对每一列，切片数据直到当前的预测点
                df_column_actual = pca_combined_df[col]
                
                # 使用历史数据进行此 PCA 组件的预测
                try:
                    # 尝试使用 ARIMA 进行预测
                    forecast_arima = arima_forecast(df_column_actual)
                except Exception as e:
                    # 如果 ARIMA 预测失败，捕获异常并打印错误
                    print(f"ARIMA forecast failed with an unexpected error: {e}")
                    print(df_column_actual)
                
                forecast_ets = ets_forecast(df_column_actual)
                forecast_stl_ets = stl_ets_forecast(df_column_actual)
                forecast_prophet = prophet_forecast(df_column_actual)
                forecast_holt_winters = holt_winters_forecast(df_column_actual)
                
                # 对该 PCA 组件的所有预测结果进行平均
                avg_forecast = np.mean([
                    forecast_arima,
                    forecast_ets,
                    forecast_stl_ets,
                    forecast_prophet,
                    forecast_holt_winters
                ])
                forecast_avg.append(avg_forecast)

            # 使用 OLS 模型预测 y 值
            forecast_y_scaled = model_ols.predict(forecast_avg)
            forecast_y = scaler_y.inverse_transform(forecast_y_scaled.reshape(-1, 1))

            # 将预测结果保存到列表中
            month_2_forecasts.append(forecast_y[0][0])
            month_2_dates.append(test_month)
            fitted_y_across_time.append(forecast_y[0][0])
            
            # 在图表中绘制预测的 y 值
            plt.scatter(test_month, forecast_y[0][0], color='green', label=f'Forecasted y for next quarter ({test_month.strftime("%Y-%m")})')

        elif month_in_quarter == 3:
            # 第三个月：对过去三个月的值进行平均
            df1_train_updated = df1_quarterly_all.loc['2006-12-31':test_month]  # 获取所有数据，确保获取最新值？
            
            # 更新训练结束日期为当前季度的日期
            updated_training_end_date = test_month
            
            # 将quarter_x_merged与df1_quarterly_all连接，但只选择x1_columns和x2_columns
            # 打印更新后的训练数据框
            df2_train_updated = df2.loc['2006-12-31':test_month]
            
            # 重新标准化并重新拟合模型
            X1_train_scaled = scaler_x1.fit_transform(df1_train_updated[x1_columns].ffill())
            X2_train_scaled = scaler_x2.fit_transform(df1_train_updated[x2_columns].ffill())
            y_train_scaled = scaler_y.fit_transform(df2_train_updated.values.reshape(-1, 1))
            
            # 重新计算PCA
            pca_x1_opt, X1_train_pca, d_x1 = apply_pca(X1_train_scaled)
            pca_x2_opt, X2_train_pca, d_x2 = apply_pca(X2_train_scaled)

            # 合并PCA组件
            pca_df = pd.concat([
                pd.DataFrame(X1_train_pca, columns=[f'X1_PC{i+1}' for i in range(d_x1)], index=df1_train_updated.index),
                pd.DataFrame(X2_train_pca, columns=[f'X2_PC{i+1}' for i in range(d_x2)], index=df1_train_updated.index)
            ], axis=1)

            # 选择其他列数据
            other_columns = [col for col in df1_train_updated.columns if col not in x1_columns + x2_columns]
            other_data = df1_train_updated[other_columns].dropna(axis=1)  # 删除包含NaN值的列
            
            # 合并PCA数据和其他数据
            X_train = pd.concat([pca_df, other_data], axis=1)
            
            # 重新拟合OLS回归模型
            model_ols = sm.OLS(y_train_scaled, X_train).fit()
            
            # 修正PCA/OLS格式
            forecast_avg = []
            
            df = X_train

            for col in X_train.columns:
                # 对每列，切片数据直到当前预测点
                df_column_actual = X_train[col]
                
                # 对此PCA组件进行预测，使用历史数据
                forecast_arima = arima_forecast(df_column_actual)
                forecast_ets = ets_forecast(df_column_actual)
                forecast_stl_ets = stl_ets_forecast(df_column_actual)
                forecast_prophet = prophet_forecast(df_column_actual)
                forecast_holt_winters = holt_winters_forecast(df_column_actual)
                
                # 对此PCA组件的预测结果进行平均
                avg_forecast = np.mean([
                    forecast_arima,
                    forecast_ets,
                    forecast_stl_ets,
                    forecast_prophet,
                    forecast_holt_winters
                ])
                forecast_avg.append(avg_forecast)

            forecast_y_scaled = model_ols.predict(forecast_avg)
            forecast_y = scaler_y.inverse_transform(forecast_y_scaled.reshape(-1, 1))

            # 将预测值添加到相应的列表中
            month_3_forecasts.append(forecast_y[0][0])
            month_3_dates.append(test_month)
            fitted_y_across_time.append(forecast_y[0][0])

            # 计算预测值与实际值之间的误差
            actual_y = df2.loc[test_month].values[0]  # 获取当前季度的实际y值
            error = forecast_y[0][0] - actual_y

            # 将月1、月2和月3的预测误差分别添加到相应的列表中
            if month_in_quarter == 1:
                month_1_errors.append(error)
            elif month_in_quarter == 2:
                month_2_errors.append(error)
            elif month_in_quarter == 3:
                month_3_errors.append(error)

            # 绘制预测值
            plt.scatter(test_month, forecast_y[0][0], color='orange', label=f'预测的下季度y值 ({test_month.strftime("%Y-%m")})')
            
            # 在重新拟合模型后，绘制所有拟合值，直到当前测试月份
            # fitted_y_all_scaled = model_ols.fittedvalues
            # fitted_y_all = scaler_y.inverse_transform(fitted_y_all_scaled.reshape(-1, 1))
            # plt.plot(df2.loc['2006-12-31':test_month].index, fitted_y_all, color=cmap(i % 10), linestyle='--', label=f'拟合y值（截至{test_month.strftime("%Y-%m")}）')

            training_end_date = updated_training_end_date

    # 设置测试期间的时间范围
    start_date = pd.to_datetime(f'{testing_start_date}')
    end_date = pd.to_datetime('2024-07-30')
    plt.xlim([start_date, end_date])
    plt.ylim([-10, 60])

    # 设置x轴为按月显示
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 绘制月1、月2和月3的预测值
    plt.plot(month_1_dates, month_1_forecasts, color='blue', label='月1预测值')
    plt.plot(month_2_dates, month_2_forecasts, color='green', label='月2预测值')
    plt.plot(month_3_dates, month_3_forecasts, color='orange', label='月3预测值')

    # 获取预测月份对应的实际y值
    shifted_date = pd.to_datetime(testing_start_date) + pd.DateOffset(months=5)
    actual_y_values = df2.loc[shifted_date:].values.reshape(-1)

    # 计算每个月份的准确性
    month_1_mae, month_1_rmse = calculate_accuracy(month_1_forecasts[:len(actual_y_values)], actual_y_values)
    month_2_mae, month_2_rmse = calculate_accuracy(month_2_forecasts[:len(actual_y_values)], actual_y_values)
    month_3_mae, month_3_rmse = calculate_accuracy(month_3_forecasts[:len(actual_y_values)], actual_y_values)

    # 打印每个月份的预测准确性结果
    print(f"月1预测准确性: MAE = {month_1_mae}, RMSE = {month_1_rmse}")
    print(f"月2预测准确性: MAE = {month_2_mae}, RMSE = {month_2_rmse}")
    print(f"月3预测准确性: MAE = {month_3_mae}, RMSE = {month_3_rmse}")

    # 打印最后三个月的预测结果
    print("最后三个月1预测值:", month_1_forecasts[-3:])
    print("最后三个月2预测值:", month_2_forecasts[-3:])
    print("最后三个月3预测值:", month_3_forecasts[-3:])

    # 旋转x轴标签，确保可读性
    plt.xticks(rotation=45)

    # 添加标签和标题
    plt.xlabel('日期')
    plt.ylabel('y值')
    plt.title('实际与拟合y值')

    # 显示图表
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 创建第二个图形，显示将预测点向前平移一个季度
    # -------------------------------------------------

    # 将预测日期向前平移一个季度（3个月）
    shifted_month_1_dates = [d + pd.DateOffset(months=5) for d in month_1_dates]  # 第一个月平移5个月
    shifted_month_2_dates = [d + pd.DateOffset(months=4) for d in month_2_dates]  # 第二个月平移4个月
    shifted_month_3_dates = [d + pd.DateOffset(months=3) for d in month_3_dates]  # 第三个月平移3个月

    # 创建新的图形用于显示平移后的预测
    plt.figure(figsize=(12, 8))

    # 绘制实际的y值（2006-12-31到2024-06-30）
    plt.plot(df2.loc['2006-12-31':testing_end_date].index, full_actual_y, label='实际y值', linestyle='--', color='blue')

    # 设置x轴以显示按月间隔的时间
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 绘制平移后的预测点
    plt.plot(shifted_month_1_dates, month_1_forecasts, color='blue', label='平移后的第1个月预测')
    plt.plot(shifted_month_2_dates, month_2_forecasts, color='green', label='平移后的第2个月预测')
    plt.plot(shifted_month_3_dates, month_3_forecasts, color='orange', label='平移后的第3个月预测')

    # 为每个预测点添加散点图，以突出显示预测点
    plt.scatter(shifted_month_1_dates, month_1_forecasts, color='blue')
    plt.scatter(shifted_month_2_dates, month_2_forecasts, color='green')
    plt.scatter(shifted_month_3_dates, month_3_forecasts, color='orange')

    # 为第二个图形添加标签和标题
    plt.xlabel('日期')
    plt.ylabel('y值')
    plt.title('将预测y值向前平移一个季度')

    # 对平移后的图形进行缩放，聚焦测试期间
    plt.xlim([start_date + pd.DateOffset(months=3), end_date + pd.DateOffset(months=3)])
    plt.ylim([-10, 60])

    # 旋转x轴标签，便于阅读
    plt.xticks(rotation=45)

    # 添加图例
    plt.legend()

    # 显示平移后的预测图
    plt.tight_layout()
    plt.show()

    # 将预测日期按指定的月份进行平移
    # 将full_actual_y转换为1维Series，并与对应的索引匹配
    actual_y_series = pd.Series(full_actual_y.ravel(), index=df2.loc['2006-12-31':testing_end_date].index)

    # 平移第一个月的预测日期
    shifted_month_1_dates = [d + pd.DateOffset(months=5) for d in month_1_dates]

    # 将预测数组转换为以平移后的日期为索引的pandas Series
    shifted_month_1_series = pd.Series(month_1_forecasts, index=shifted_month_1_dates)
    shifted_month_2_series = pd.Series(month_2_forecasts, index=shifted_month_1_dates)
    shifted_month_3_series = pd.Series(month_3_forecasts, index=shifted_month_1_dates)

    # 对齐预测值与实际y值，找到共有的日期
    common_dates_1 = shifted_month_1_series.index.intersection(actual_y_series.index)
    common_dates_2 = shifted_month_2_series.index.intersection(actual_y_series.index)
    common_dates_3 = shifted_month_3_series.index.intersection(actual_y_series.index)

    # 仅保留与实际y值匹配的日期，重新索引预测系列
    aligned_month_1_forecasts = shifted_month_1_series.reindex(common_dates_1)
    aligned_month_2_forecasts = shifted_month_2_series.reindex(common_dates_2)
    aligned_month_3_forecasts = shifted_month_3_series.reindex(common_dates_3)

    # 重新索引实际y系列，匹配每个月的共同日期
    aligned_actual_y_1 = actual_y_series.reindex(common_dates_1)
    aligned_actual_y_2 = actual_y_series.reindex(common_dates_2)
    aligned_actual_y_3 = actual_y_series.reindex(common_dates_3)

    # 计算实际y值和预测值之间的变化（差异）
    actual_y_changes_1 = aligned_actual_y_1.diff().dropna()
    actual_y_changes_2 = aligned_actual_y_2.diff().dropna()
    actual_y_changes_3 = aligned_actual_y_3.diff().dropna()

    forecast_changes_1 = aligned_month_1_forecasts.diff().dropna()
    forecast_changes_2 = aligned_month_2_forecasts.diff().dropna()
    forecast_changes_3 = aligned_month_3_forecasts.diff().dropna()

    # 对实际y值和预测y值进行方向分类
    actual_directions_1 = actual_y_changes_1.apply(categorize_direction)
    actual_directions_2 = actual_y_changes_2.apply(categorize_direction)
    actual_directions_3 = actual_y_changes_3.apply(categorize_direction)

    forecast_directions_1 = forecast_changes_1.apply(categorize_direction)
    forecast_directions_2 = forecast_changes_2.apply(categorize_direction)
    forecast_directions_3 = forecast_changes_3.apply(categorize_direction)

    # 比较实际方向与预测方向，计算方向准确性
    def calculate_direction_accuracy(actual, forecast):
        correct_directions = (actual == forecast)  # 如果预测方向与实际一致
        accuracy = correct_directions.sum() / len(correct_directions) * 100  # 计算准确率
        return accuracy

    # 计算每个月份的方向准确性
    month_1_accuracy = calculate_direction_accuracy(actual_directions_1[1:], forecast_directions_1[1:])
    month_2_accuracy = calculate_direction_accuracy(actual_directions_2[1:], forecast_directions_2[1:])
    month_3_accuracy = calculate_direction_accuracy(actual_directions_3[1:], forecast_directions_3[1:])

    # 打印方向准确性结果
    print(f"第1个月的方向准确性: {month_1_accuracy:.2f}%")
    print(f"第2个月的方向准确性: {month_2_accuracy:.2f}%")
    print(f"第3个月的方向准确性: {month_3_accuracy:.2f}%")

    # 可选：打印最后几个预测方向和实际方向，以便可视化检查
    comparison_df = pd.DataFrame({
        '实际方向 (第1个月)': actual_directions_1,
        '预测方向 (第1个月)': forecast_directions_1,
        '实际方向 (第2个月)': actual_directions_2,
        '预测方向 (第2个月)': forecast_directions_2,
        '实际方向 (第3个月)': actual_directions_3,
        '预测方向 (第3个月)': forecast_directions_3
    })
    print(comparison_df.tail(10))  # 显示最后10个时段进行对比

    # 提示：可以进一步添加更多前瞻性指标 - 及时调整和因子优化

