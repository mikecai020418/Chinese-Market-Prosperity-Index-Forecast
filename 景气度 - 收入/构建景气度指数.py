'''

# 最后决定使用因子：

格兰杰因果关系：

弱前瞻：p < 0.15
强前瞻：p < 0.05

## 宏观

Shibor（利率 - 强即时）
国家一般公共收入（政府收入 - 强前瞻）
税收收入：累计同比（税收 - 强前瞻）
社会零售品消费总额（消费 - 弱即时）
税收收入：关税（进口 - 强即时）

## 中观

**工业产量**：
机械产量：水泥专用设备（强前瞻）
电力：全社会用电量（强即时性因子）
交流电动机（弱前瞻因子）

**工业利润**：
亏损企业亏损总额（强即时），工业企业利润（强前瞻）

**经济活跃度**：
货邮周转量（弱前瞻）

'''

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

if __name__ == "__main__":
    # 加载并准备 df1 和 df2 数据
    df1, column_frequency = load_and_prepare_df1(file_path)
    df2 = load_and_prepare_df2(final_df.copy())

    # 根据列频率对数据进行时间偏移
    for col, freq in column_frequency.items():
        shift_data_by_frequency(df1, col, freq)
    
    """1. 完整收入景气度指数"""
    # 对齐 df1 和 df2 的日期范围

    # 2006-06-30 - 64.29% acuuracy
    #使用：2006-12-31 - 65.22% accuracy
    # 2007-12-31 - 66.15%
    # 2008-03-31 - 67.19%
    # '2010-06-30' - 67.26%
    # '2010-09-30' - 68.52%
    # '2018-12-31' - 76.19%
    start_date = '2010-09-30'
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

    # 绘制 x1 的解释方差图
    # plt.figure(figsize=(8, 5))
    # plt.bar(range(1, len(explained_variance_x1) + 1), explained_variance_x1, alpha=0.7, align='center', label='Explained Variance')
    # plt.step(range(1, len(cumulative_variance_x1) + 1), cumulative_variance_x1, where='mid', label='Cumulative Variance')
    # plt.ylabel('Explained Variance Ratio')
    # plt.xlabel('Principal Components')
    # plt.title('Explained Variance of PCA for x1')
    # plt.legend(loc='best')
    # plt.show()

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

    # 绘制 x2 的解释方差图
    # plt.figure(figsize=(8, 5))
    # plt.bar(range(1, len(explained_variance_x2) + 1), explained_variance_x2, alpha=0.7, align='center', label='Explained Variance')
    # plt.step(range(1, len(cumulative_variance_x2) + 1), cumulative_variance_x2, where='mid', label='Cumulative Variance')
    # plt.ylabel('Explained Variance Ratio')
    # plt.xlabel('Principal Components')
    # plt.title('Explained Variance of PCA for x2')
    # plt.legend(loc='best')
    # plt.show()

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

    # 输出回归的总结
    print(model_ols.summary())  # 基本拟合 - 包括其他因素以提高拟合度

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
    plt.ylabel('上证指数总营业收入同比（%)')
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

    """2. 仅前瞻收入景气度指数"""

    # 定义选择的列，包含经济和工业指标因子
    selected_columns = [
        "国家一般公共收入:合计:累计同比",  # 国家一般公共收入（政府收入 - 强前瞻）
        "税收收入:累计同比",               # 累计同比（税收 - 强前瞻）
        "水泥专用设备:产量:当月同比",       # 机械产量：水泥专用设备（强前瞻）
        "交流电动机:产量:当月同比",         # 交流电动机（弱前瞻因子）
        "规模以上工业企业:利润总额:累计同比", # 工业企业利润（强前瞻）
        "货邮周转量:当月同比"              # 货邮周转量（弱前瞻）
    ]

    # 从df1中提取选择的列
    df_selected = df1[selected_columns]

    # 打印提取的列，检查数据
    print(df_selected.head())

    # 使用PCA因子（来自x1和x2）以及目标变量y进行回归分析

    # 步骤1：从df1_quarterly_all中识别不属于x1_columns或x2_columns的列

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
    plt.ylabel('上证指数总营业收入同比（%)')
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

    '''3.平均收入景气度指数'''

    # 将fitted_values_2转换为NumPy数组并调整为列向量形式
    fitted_values_2 = fitted_values_2.to_numpy().reshape(-1, 1)

    # 计算fitted_values_1和fitted_values_2的平均值
    fitted_values_mean = (fitted_values + fitted_values_2) / 2

    # 绘制实际值与拟合均值的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, y_true, label='实际值', color='blue')  # 绘制实际值
    plt.plot(df1.index, fitted_values_mean, label='拟合值', color='red', linestyle='--')  # 绘制拟合值
    plt.title('拟合值与实际值对比')  # 设置图表标题
    plt.xlabel('日期')  # 设置x轴标签
    plt.ylabel('上证指数总营业收入同比（%)')  # 设置y轴标签
    plt.legend()  # 添加图例

    # 设置x轴主要刻度为每8个月
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))  # 设置主要刻度为每8个月
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化x轴刻度为年-月

    # 旋转x轴标签并调整图表布局
    plt.xticks(rotation=45)  # 旋转x轴标签
    plt.tight_layout()  # 调整布局以防标签重叠

    # 显示图表
    plt.show()

    # 对y_true和fitted_values_mean计算方向准确率
    y_true_values = y_true.flatten()  # 将实际值转换为一维数组
    fitted_values_mean_array = fitted_values_mean  # 平均拟合值

    # 打印方向准确率
    accuracy_percentage = direction_accuracy(y_true_values, fitted_values_mean_array)
    print(f"方向准确率: {accuracy_percentage:.2f}%")


