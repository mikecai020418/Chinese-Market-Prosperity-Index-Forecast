'''
# 最后决定使用因子：

## 宏观

CRB(通胀因子 - 前瞻）
工业GDP(经济增长-即时）
房地产开发投资完成额（投资 - 前瞻）
国家一般公共收入（政府收入 - 弱前瞻）
社会零售品（消费 - 弱前瞻）
税收收入：关税（进口 - 前瞻）

## 中观

**工业产量**：
机械产量：交流电动机（强即时）
金属集装箱（强即时）
水泥专用设备（强前瞻）
钢铁产量：企业景气指数（强即时因子）
电力：全社会用电量（弱前瞻性因子）

**工业利润**：
工业增加值（强即时）
工业企业利润（强即时）

**经济活跃度**：
货邮周转量（即时）

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'Arial Unicode MS'
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

def plot_explained_variance(explained_variance, cumulative_variance, title):
    """
    绘制PCA主成分解释方差条形图和累积解释方差折线图。
    
    参数:
        explained_variance (ndarray): 每个主成分的解释方差。
        cumulative_variance (ndarray): 累积解释方差。
        title (str): 图表标题。
    """
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Explained Variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

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
    print(f'Number of components for x1: {d_x1}')
    plot_explained_variance(explained_variance_x1, cumulative_variance_x1, 'Explained Variance of PCA for x1')
    
    # 对x2进行PCA分析
    d_x2, pca_x2, pca_x2_components, explained_variance_x2, cumulative_variance_x2 = pca_analysis(X2_scaled, variance_threshold=0.9)
    print(f'Number of components for x2: {d_x2}')
    plot_explained_variance(explained_variance_x2, cumulative_variance_x2, 'Explained Variance of PCA for x2')

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
    
    # 创建时间索引，并绘制实际值与拟合值对比图
    date_range = pd.date_range(start='2006-03-31', end='2024-06-30', periods=len(y_true))
    plt.figure(figsize=(10, 6))
    plt.plot(date_range, y_true, label='Actual Values', color='blue')
    plt.plot(date_range, fitted_values, label='Fitted Values', color='red', linestyle='--')
    plt.title('Fitted Values vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('上证指数净利润同比（%）')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 计算并打印方向准确度（再次验证）
    accuracy = direction_accuracy(y_true.flatten(), fitted_values.flatten())
    print(f"Directional Accuracy: {accuracy:.2f}%")
    
    # # 可选：保存PCA结果和回归摘要
    # pca_df.to_excel('pca_results.xlsx')


    # 0.796 R^2/ 69.44 % directional accuaracy

    """2. 仅前瞻利润景气度指数"""

    # 定义选择的列，包含经济和工业指标因子
    selected_columns = [
        'CRB现货指数:工业', 
        '房地产开发投资完成额:建筑工程:累计同比',               
        '国家一般公共收入:合计:累计同比',       
        '社会消费品零售总额:累计同比',         
        '税收收入:关税:累计同比', 
        '水泥专用设备:产量:当月同比',
        '全社会用电量:累计同比'              
    ]

    df1 = df1_quarterly_all
    df_selected = df1[selected_columns]

    # 步骤2：过滤出在df1的日期范围内不含NaN值的列
    valid_other_columns = []
    for col in df_selected.columns:
        # 检查列是否在df1日期范围内存在NaN值
        if df1[col].loc[df1.index].isna().sum() == 0:
            valid_other_columns.append(col)
        else:
            print(f"列 '{col}' 在df1的日期范围内包含NaN值，将被移除。")
    
        # 步骤3：从df1_quarterly_all提取有效的附加列
    other_data = df1_quarterly_all[valid_other_columns].loc[df1.index]
        # 使用df2的数据对齐日期范围，并设置目标变量
    y_true = np.array(df2.reindex(df1.index, method='ffill')).reshape(-1, 1)

        # 使用OLS模型拟合其他数据与目标变量的关系
    model_ols = sm.OLS(y_true, other_data).fit()

    # 打印回归模型的摘要信息
    print(model_ols.summary())  # 包含额外因子以提高拟合效果
    # 提取模型的拟合值
    fitted_values_2 = model_ols.fittedvalues  # 拟合值（已缩放形式）

        # 绘制拟合值与实际值的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, y_true, label='实际值', color='blue')
    plt.plot(df1.index, fitted_values_2, label='拟合值', color='red', linestyle='--')
    plt.title('拟合值与实际值对比')
    plt.xlabel('日期')
    plt.ylabel('上证指数归母净利润同比（%)')
    plt.legend()

    # 设置x轴主要刻度为每8个月
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))  # 设置主要刻度为每8个月
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化x轴刻度为年-月

    # 旋转x轴标签，并调整布局
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示图表
    plt.show()

    # 对y_true和拟合值fitted_values_2应用方向准确率计算
    y_true_values = y_true.flatten()  # 确保为一维数组
    fitted_values_2_array = fitted_values_2

    # 打印方向准确率
    accuracy_percentage = direction_accuracy(y_true_values, fitted_values_2_array)
    print(f"方向准确率: {accuracy_percentage:.2f}%")

    # 0.823 R^2/ 70.83 % directional accuaracy




