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

# FIx: include other variables, not just the pca items

# 设置中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

final_df = pd.read_csv('final_df.csv')
# ============================== 数据加载与清洗 ==============================
def load_and_clean_data(file_path):
    """
    载入Excel文件数据，转换日期格式，将日期设置为索引，并对所有列进行清洗（移除逗号并转换为数值型）。
    
    参数:
        file_path (str): Excel文件的路径。
        
    返回:
        pd.DataFrame: 清洗后的数据框。
    """
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    invalid_dates = df['Date'].isna().sum()
    print(f"Number of invalid date entries: {invalid_dates}")
    df.set_index('Date', inplace=True)
    return df.apply(clean_column)

def clean_column(col):
    """
    清洗数据列：如果列数据为字符串，则移除逗号并转换为数值；否则直接转换为数值。
    
    参数:
        col (pd.Series): 数据列。
        
    返回:
        pd.Series: 清洗后的数据列。
    """
    if col.dtype == 'object':
        return pd.to_numeric(col.str.replace(',', '', regex=False), errors='coerce')
    return pd.to_numeric(col, errors='coerce')

# ============================== 数据频率调整 ==============================
def shift_data_by_frequency(df, col, freq_type, current_date_limit):
    """
    根据指定的频率对数据进行位移，并将超出current_date_limit的部分设置为NaN。
    
    参数:
        df (pd.DataFrame): 数据框。
        col (str): 要位移的列名。
        freq_type (str): 数据频率，'D'表示日，'M'表示月，'Q'表示季度。
        current_date_limit (pd.Timestamp): 超出此日期的数据将被忽略。
        
    返回:
        pd.Series: 位移后的数据列。
    """
    if freq_type == 'D':
        return df[col].shift(-1)
    elif freq_type == 'M':
        next_month = df.index + pd.DateOffset(months=1)
        next_month_10th = next_month + pd.offsets.Day(7)
        shifted_data = df[col].shift(-8)
        shifted_data[next_month_10th > current_date_limit] = np.nan
        return shifted_data
    elif freq_type == 'Q':
        next_month = df.index + pd.DateOffset(months=1)
        next_month_20th = next_month + pd.offsets.Day(14)
        shifted_data = df[col].shift(-15)
        shifted_data[next_month_20th > current_date_limit] = np.nan
        return shifted_data

def apply_shifts(df, column_frequency, current_date_limit):
    """
    遍历列频率映射，对数据框中的每个指定列应用数据位移。
    
    参数:
        df (pd.DataFrame): 数据框。
        column_frequency (dict): 列名到频率的映射字典。
        current_date_limit (pd.Timestamp): 当前日期限制。
        
    返回:
        pd.DataFrame: 应用位移后的数据框。
    """
    for col, freq in column_frequency.items():
        df[col] = shift_data_by_frequency(df, col, freq, current_date_limit)
    return df

# Function for ARIMA forecast
def arima_forecast(df_column):
    model = pm.auto_arima(df_column, seasonal=False, trace=True)
    forecast, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return forecast[0]

# Function for ETS forecast
def ets_forecast(df_column):
    model = ExponentialSmoothing(df_column, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(1)
    return forecast[0]

# Function for STL-ETS forecast
def stl_ets_forecast(df_column):
    stl = STL(df_column, seasonal=13)
    result = stl.fit()
    model = ExponentialSmoothing(result.trend.dropna(), trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(1)
    return forecast[0]

# Function for Prophet forecast
def prophet_forecast(df_column):
    df_prophet = pd.DataFrame({'ds': pca_df.index, 'y': df_column})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=1, freq='Q')
    forecast = model.predict(future)
    forecast_value = forecast.iloc[-1]['yhat']
    return forecast_value

# Function for Holt-Winters forecast (triple exponential smoothing)
def holt_winters_forecast(df_column):
    model = ExponentialSmoothing(df_column, trend='add', seasonal='add', seasonal_periods=4)
    model_fit = model.fit()
    forecast = model_fit.forecast(1)
    return forecast[0]

# ============================== 数据预处理 ==============================
def preprocess_data(df1_path, df, current_date_limit): #change this
    """
    载入并预处理数据。载入df1、清洗数据、按频率调整；载入df2并转换日期。
    同时按照指定日期范围过滤，并将df1的索引重置为与df2一致（前向填充缺失值）。
    
    参数:
        df1_path (str): 第一个数据集的文件路径。
        df2 (pd.DataFrame): 第二个数据集。
        current_date_limit (pd.Timestamp): 当前日期限制。
        
    返回:
        tuple: (df1, df2, column_frequency)
    """
    df1 = load_and_clean_data(df1_path)
    
    # 定义各列数据频率映射
    column_frequency = {
        '房地产开发投资完成额:建筑工程:累计同比': 'M',
        'CRB现货指数:工业': 'D',
        'GDP:不变价:第二产业:工业:当季同比': 'Q',
        '税收收入:关税:累计同比': 'Q',
        '国家一般公共收入:合计:累计同比': 'M',
        '社会消费品零售总额:累计同比': 'M',
        '交流电动机:产量:当月同比': 'M',
        '金属集装箱:产量:当月同比': 'M',
        '水泥专用设备:产量:当月同比': 'M',
        '企业景气指数:制造业:黑色金属冶炼及压延加工业': 'Q',
        '全社会用电量:累计同比': 'M',
        '规模以上工业增加值:国有控股企业:当月同比': 'M',
        '规模以上工业企业:利润总额:累计同比': 'M',
        '货邮周转量:当月同比': 'M'
    }
    
    df1 = apply_shifts(df1, column_frequency, current_date_limit)
    # 载入df2数据，并转换日期格式
    df2 = df.copy()
    df2 = df2.drop(columns = 'Unnamed: 0')
    df2['date'] = pd.to_datetime(df2['date'])
    df2 = df2.set_index('date')
    # 过滤指定日期区间
    start_date = '2006-03-31'
    end_date = '2024-06-30'
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]
    
    return df1, df2, column_frequency

# ============================== 季度数据处理 ==============================
def process_quarterly_and_accumulated_data(df, column, df2):
    """
    处理df中指定列的季度或“累计同比”数据，将数据与df2的日期对齐，并前向填充缺失值。
    
    参数:
        df (pd.DataFrame): 数据框1。
        column (str): 要处理的列名。
        df2 (pd.DataFrame): 数据框2，用于对齐日期。
        
    返回:
        pd.DataFrame: 处理后的季度数据。
    """
    df_clean = df[[column]].dropna()
    earliest_valid_date = df_clean.index.min()
    df2_truncated = df2[df2.index >= earliest_valid_date]
    return df_clean.reindex(df2_truncated.index, method='ffill')

def build_quarterly_dataframe(df1, df2, column_frequency):
    """
    根据列频率处理df1中的数据，并构建一个季度频率的DataFrame，
    包含季度数据和累计同比数据（对于月/日数据，进行重采样）。
    
    参数:
        df1 (pd.DataFrame): 预处理后的数据框1。
        df2 (pd.DataFrame): 数据框2，用于对齐日期。
        column_frequency (dict): 各列数据频率映射。
        
    返回:
        pd.DataFrame: 季度频率数据框。
    """
    df1_quarterly_all = pd.DataFrame()
    
    # 处理季度和累计同比数据
    for col in df1.columns:
        if column_frequency[col] == 'Q' or "累计同比" in col:
            df1_quarterly = process_quarterly_and_accumulated_data(df1, col, df2)
            df1_quarterly_all = pd.concat([df1_quarterly_all, df1_quarterly], axis=1)

    # 处理月度和日度数据（不包含“累计同比”）
    for col, freq in column_frequency.items():
        if freq == 'M' and "累计同比" not in col:
            df_temp = df1[col].resample('Q').mean()
            df1_quarterly_all[col] = df_temp.reindex(df2.index, method='ffill')
        elif freq == 'D' and "累计同比" not in col:
            df_temp = df1[col].resample('Q').mean()
            df1_quarterly_all[col] = df_temp.reindex(df2.index, method='ffill')
    
    # 过滤最终数据区间
    start_date = '2001-12-31'
    end_date = '2024-06-30'
    df1_quarterly_all = df1_quarterly_all.loc[start_date:end_date]
    
    # 导出结果到Excel
    df1_quarterly_all.to_excel('全部因子—季度频率.xlsx')
    print("Processed quarterly data has been saved as '全部因子_季度频率.xlsx'.")
    
    return df1_quarterly_all

# ============================== 标准化与PCA分析 ==============================
def standardize_data(x1, x2, df2):
    """
    对x1、x2和目标变量y（df2）进行标准化处理。
    
    参数:
        x1 (pd.DataFrame): 特征集1。
        x2 (pd.DataFrame): 特征集2。
        df2 (pd.DataFrame): 目标变量数据框。
        
    返回:
        tuple: (X1_scaled, X2_scaled, y_scaled, scaler_x1, scaler_x2, scaler_y)
    """
    scaler_x1 = StandardScaler()
    X1_scaled = scaler_x1.fit_transform(x1)
    
    scaler_x2 = StandardScaler()
    X2_scaled = scaler_x2.fit_transform(x2)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(np.array(df2).reshape(-1, 1))
    
    return X1_scaled, X2_scaled, y_scaled, scaler_x1, scaler_x2, scaler_y

def pca_analysis(X_scaled, variance_threshold=0.9):
    """
    对标准化后的数据执行PCA分析，返回最佳主成分数量、PCA模型、转换后的组件、
    每个主成分的解释方差以及累积解释方差。
    
    参数:
        X_scaled (ndarray): 标准化后的数据。
        variance_threshold (float): 累积解释方差阈值（默认0.9）。
        
    返回:
        tuple: (最佳主成分数量, PCA模型, 主成分数组, 解释方差, 累积解释方差)
    """
    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    d = np.argmax(cumulative_variance > variance_threshold) + 1
    pca_opt = PCA(n_components=d)
    components = pca_opt.fit_transform(X_scaled)
    return d, pca_opt, components, explained_variance, cumulative_variance

# ============================== 回归分析与预测准确度 ==============================
def regression_analysis(y_scaled, pca_components_combined):
    """
    使用OLS对目标变量y和合并后的PCA组件进行回归分析。
    
    参数:
        y_scaled (ndarray): 标准化后的目标变量。
        pca_components_combined (ndarray): 合并后的PCA组件。
        
    返回:
        回归模型对象
    """
    model = sm.OLS(y_scaled, pca_components_combined).fit()
    return model

# Function to calculate forecasted y based on OLS regression
def calculate_forecast_y(forecast_avg):
    # Step 1: Convert forecast_avg to a NumPy array and reshape it for prediction
    forecast_avg = np.array(forecast_avg).reshape(1, -1)  # Ensure it has the correct shape for model prediction

    # Step 2: Calculate forecast_y_scaled using model_ols
    forecast_y_scaled = model_ols.predict(forecast_avg)[0] 

    # Step 3: Calculate confidence intervals using get_prediction()
    prediction_results = model_ols.get_prediction(forecast_avg)
    conf_int_scaled = prediction_results.conf_int()  # Provides the lower and upper bounds

    # Step 4: Extract lower and upper bounds for confidence interval
    conf_interval_low_scaled, conf_interval_high_scaled = conf_int_scaled[0]
    
    # Inverse transform to the original y
    forecast_y = scaler_y.inverse_transform([[forecast_y_scaled]])[0][0]
    conf_interval_low = scaler_y.inverse_transform([[conf_interval_low_scaled]])[0][0]
    conf_interval_high = scaler_y.inverse_transform([[conf_interval_high_scaled]])[0][0]
    
    return forecast_y, conf_interval_low, conf_interval_high

def forecast_column(df, col):
    """Applies different forecasting models to a given column and returns the forecasts."""
    forecast_methods = {
        'ARIMA': arima_forecast,
        'ETS': ets_forecast,
        'STL-ETS': stl_ets_forecast,
        'Prophet': prophet_forecast,
        'Holt-Winters': holt_winters_forecast
    }
    forecasts = {name: method(df[col]) for name, method in forecast_methods.items()}
    return forecasts

def plot_forecasts(df, col, forecasts, forecast_date=pd.Timestamp('2024-09-30')):
    """Plots the actual data and forecasts from different models."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[col], label='Actual')
    plt.axvline(pd.Timestamp('2024-06-30'), color='gray', linestyle='--')  # Mark current date
    
    colors = {'ARIMA': 'red', 'ETS': 'green', 'STL-ETS': 'purple', 'Prophet': 'orange', 'Holt-Winters': 'blue'}
    for model, forecast in forecasts.items():
        if isinstance(forecast, (list, np.ndarray)):  # If forecast is an array
            forecast = forecast[-1]  # Take the last value (or adjust as needed)
        print(f"{model} forecast type: {type(forecast)}, value: {forecast}")
        plt.scatter(forecast_date, forecast, color=colors[model], label=f'Forecast ({model})')

    plt.title(f'Forecast for {col}')
    plt.legend()
    plt.show()

def inverse_transform_y(scaler_y, y_scaled, model_ols):
    """Inverse transforms the scaled y and extracts fitted values."""
    y_true = scaler_y.inverse_transform(y_scaled)
    fitted_values = scaler_y.inverse_transform(model_ols.fittedvalues.reshape(-1, 1))
    return y_true, fitted_values

def calculate_avg_forecast(forecast_results, columns):
    """Calculates the average forecast from multiple models."""
    return [np.mean(list(forecast_results[col].values())) for col in columns]

def plot_y_forecast(historical_dates, historical_y_true, fitted_values, forecast_y, conf_int_low, conf_int_high):
    """Plots historical y, fitted values, forecast, and confidence interval."""
    last_hist_date, last_hist_value = historical_dates[-1], float(historical_y_true[-1])
    forecast_date = pd.Timestamp('2024-09-30')

    plt.figure(figsize=(10, 6))
    plt.plot(historical_dates, historical_y_true, label='Actual y (True)', color='blue')
    plt.plot(historical_dates, fitted_values, label='Fitted y (OLS)', color='green', linestyle='--')

    plt.fill_between([last_hist_date, forecast_date], [last_hist_value, conf_int_low], 
                     [last_hist_value, conf_int_high], color='red', alpha=0.3, label='Confidence Interval')

    plt.scatter(forecast_date, forecast_y, color='red', label='Forecasted y', zorder=5)
    plt.plot([last_hist_date, forecast_date], [last_hist_value, forecast_y], 
             color='blue', linestyle='--', label='Forecast Trend')

    plt.axvline(forecast_date, color='gray', linestyle='--', label='Forecast Date')

    plt.title('Historical and Forecasted y (with Confidence Interval)')
    plt.xlabel('Date')
    plt.ylabel('上证指数净利润同比（%）')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

def main_forecasting_pipeline(df, scaler_y, y_scaled, model_ols, columns_to_forecast):
    """Runs the entire forecasting pipeline."""
    forecast_results = {col: forecast_column(df, col) for col in columns_to_forecast}

    for col, forecasts in forecast_results.items():
        print(f"Forecast results for {col}:")
        for model, forecast in forecasts.items():
            print(f"{model}: {forecast}")
        plot_forecasts(df, col, forecasts)

    y_true, fitted_values = inverse_transform_y(scaler_y, y_scaled, model_ols)
    forecast_avg = calculate_avg_forecast(forecast_results, columns_to_forecast)
    
    forecast_y, conf_int_low, conf_int_high = calculate_forecast_y(forecast_avg)
    plot_y_forecast(df.index, y_true, fitted_values, forecast_y, conf_int_low, conf_int_high)

    print(f"Forecasted y for 2024-09-30: {forecast_y}")
    print(f"Last historical value on 2024-06-30: {y_true[-1]}")
    print(f"95% Confidence Interval: [{conf_int_low}, {conf_int_high}]")

def direction_accuracy(actual, fitted):
    """
    计算实际值与拟合值的方向性变化准确率（百分比）。
    
    参数:
        actual (ndarray): 实际值。
        fitted (ndarray): 拟合值。
        
    返回:
        float: 方向准确率（百分比）。
    """
    actual_diff = np.diff(actual, axis=0)
    fitted_diff = np.diff(fitted, axis=0)
    direction_match = (np.sign(actual_diff) == np.sign(fitted_diff))
    accuracy = np.mean(direction_match) * 100
    return accuracy

# 执行主函数
if __name__ == '__main__':
        # 当前日期限制
    current_date_limit = pd.Timestamp("2024-09-29")
    
    final_df = pd.read_csv('final_df.csv')
    # 载入并预处理数据（df1为全部因子集合，df2为归母净利润同比增长）
    df1, df2, column_frequency = preprocess_data('全部因子集合.xlsx', final_df, current_date_limit)
    # 处理季度数据，将df1转换为季度频率数据
    df1_quarterly_all = build_quarterly_dataframe(df1, df2, column_frequency)
    # 重置索引并确保日期列正确（此处假设已存在列 'date' 或 'index'）
    df1_quarterly_all.reset_index(inplace=True)
    # 如果重置后列名为 'index'，则将其作为日期列；否则使用 'date'
    if 'index' in df1_quarterly_all.columns:
        df1_quarterly_all.rename(columns={'index': 'date'}, inplace=True)
    df1_quarterly_all['date'] = pd.to_datetime(df1_quarterly_all['date'])
    df1_quarterly_all.set_index('date', inplace=True)
    # 再次清洗数据
    df1_quarterly_all = df1_quarterly_all.apply(clean_column)
    
    # 重新载入df2并过滤日期（归母净利润同比增长数据）
    start_date = '2006-04-01'
    end_date = '2024-06-30'
    df1_quarterly_all.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    df1_quarterly_all = df1_quarterly_all.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]
    # 删除df2中可能存在的多余列
    
    # 选择x1和x2的相关列
    x1_columns = [
        '交流电动机:产量:当月同比',  # 交流电动机产量
        '金属集装箱:产量:当月同比',  # 金属集装箱产量
        '水泥专用设备:产量:当月同比',  # 水泥专用设备产量
        '全社会用电量:累计同比'       # 全社会用电量
    ]
    x2_columns = [
        '规模以上工业增加值:国有控股企业:当月同比',  # 工业增加值
        '规模以上工业企业:利润总额:累计同比'          # 工业企业利润
    ]
    
    # 从df1中提取x1和x2数据，并进行前向填充
    x1 = df1_quarterly_all[x1_columns].fillna(method='ffill')
    x2 = df1_quarterly_all[x2_columns].fillna(method='ffill')
    
    # 标准化x1、x2以及目标变量y（df2）
    X1_scaled, X2_scaled, y_scaled, scaler_x1, scaler_x2, scaler_y = standardize_data(x1, x2, df2)
    
    # 对x1进行PCA分析
    d_x1, pca_x1, pca_x1_components, explained_variance_x1, cumulative_variance_x1 = pca_analysis(X1_scaled, variance_threshold=0.9)

    # 对x2进行PCA分析
    d_x2, pca_x2, pca_x2_components, explained_variance_x2, cumulative_variance_x2 = pca_analysis(X2_scaled, variance_threshold=0.9)

    # 将 x1 的 PCA 成分和 x2 的 PCA 成分合并
    pca_df = pd.DataFrame(pca_x1_components, columns=[f'X1_PC{i+1}' for i in range(d_x1)])
    pca_df = pd.concat([pca_df, pd.DataFrame(pca_x2_components, columns=[f'X2_PC{i+1}' for i in range(d_x2)])], axis=1)
    pca_df.index = df1_quarterly_all.index  # 保持原始索引（日期）
    globals()['pca_df'] = pca_df
    # 第一步：识别 df1_quarterly_all 中不属于 x1_columns 或 x2_columns 的列
    other_columns = [col for col in df1_quarterly_all.columns if col not in x1_columns + x2_columns]

    # 第二步：过滤出 df1 时间范围内没有 NaN 值的列
    valid_other_columns = []
    for col in other_columns:
        # 检查该列在 df1 时间范围内是否包含 NaN 值
        if df1_quarterly_all[col].loc[df1_quarterly_all.index].isna().sum() == 0:
            valid_other_columns.append(col)
        else:
            print(f"列 '{col}' 在 df1 时间范围内包含 NaN 值，将被移除。")
    # 第三步：从 df1_quarterly_all 中提取有效的附加列
    other_data = df1_quarterly_all[valid_other_columns].loc[df1_quarterly_all.index]
    globals()['other_data'] = other_data

    # 合并x1和x2的PCA组件, 将 pca_components_combined 与附加列连接
    pca_components_combined = np.concatenate([pca_x1_components, pca_x2_components, other_data], axis=1)
    globals()['pca_components_combined'] = pca_components_combined
    # 进行OLS回归分析
    model_ols = regression_analysis(y_scaled, pca_components_combined)
    print(model_ols.summary())
    
    # 将回归预测值还原为原始单位
    y_true = scaler_y.inverse_transform(y_scaled)
    fitted_values_scaled = model_ols.fittedvalues
    fitted_values = scaler_y.inverse_transform(fitted_values_scaled.reshape(-1, 1))

    # 计算方向准确度
    accuracy_percentage = direction_accuracy(y_true.flatten(), fitted_values.flatten())
    print(f"Directional Accuracy: {accuracy_percentage:.2f}%")
    
    # 将所有列名组合成一个列表
    # 从两个数据框中获取列名并合并
    column_names = list(pca_df.columns) + list(other_data.columns)

    # 创建具有指定索引和列名的数据框
    df_pca_components_combined = pd.DataFrame(pca_components_combined, index=df1_quarterly_all.index, columns=column_names)

    main_forecasting_pipeline(df_pca_components_combined, scaler_y, y_scaled, model_ols, df_pca_components_combined.columns)
    
    



