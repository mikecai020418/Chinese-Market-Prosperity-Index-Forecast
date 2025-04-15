import pandas
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import math
import random
from collections import deque
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, ccf

# 设置中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 定义一个函数，用于重命名从第三列开始的列名
def rename_column(col_name):
    """
    根据列名中的标记 "[报告期]" 和 "[报表类型]" 提取报告期信息，作为新的列名。
    
    参数：
        col_name (str): 原始列名。
    
    返回：
        str: 提取后的列名，若格式不符合预期，则返回原列名。
    """
    # 查找 "[报告期]" 和 "[报表类型]" 的索引位置
    start = col_name.find("[报告期]")
    end = col_name.find("[报表类型]")
    
    if start != -1 and end != -1 and end > start:
        # 提取 "[报告期]" 和 "[报表类型]" 之间的内容
        return col_name[start + len("[报告期]"):end].strip()
    else:
        # 如果格式不符合预期，返回原始列名
        return col_name

# 定义一个函数，将报告期转换为日期格式
def convert_to_date(col_name):
    """
    根据列名中的报告期信息（如 "一季", "中期", "三季", "年度"）将其转换为具体日期。
    
    参数：
        col_name (str): 原始列名。
    
    返回：
        pd.Timestamp: 对应的日期时间对象。
        str: 若无法匹配到预期格式，返回原列名。
    """
    if '一季' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-03-31")  # 一季度
    elif '中期' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-06-30")  # 二季度
    elif '三季' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-09-30")  # 三季度
    elif '年度' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-12-31")  # 四季度
    else:
        return col_name  # 如果不符合预期格式，则返回原列名

# 定义一个函数，计算每期净利润的同比增长率
def calculate_percentage_change(df):
    """
    计算数据框中每期净利润的同比增长率（百分比变化）。
    
    参数：
        df (pd.DataFrame): 包含多个报告期列的数据框。
    
    返回：
        list: 各期的同比增长率（百分比变化）。
    """
    percentage_changes = []
    # 遍历从第五列开始的每一列（确保可以进行同比比较）
    for i in range(4, len(df.columns)):
        prev_col = df.columns[i-4]  # 前一年对应的列
        current_col = df.columns[i]  # 当前列
        
        # 选取当前列和前一年列都有数值的行
        valid_rows = df[[prev_col, current_col]].dropna()
        
        # 计算前一年和当前期净利润总和
        prev_sum = valid_rows[prev_col].sum()
        current_sum = valid_rows[current_col].sum()
        
        # 计算同比增长率（百分比变化）
        percentage_change = ((current_sum - prev_sum) / prev_sum) * 100 if prev_sum != 0 else None
        percentage_changes.append(percentage_change)
    
    return percentage_changes

# 将计算结果导出为一个新的 DataFrame
def create_result_df(periods, percentage_changes):
    """
    创建包含日期和同比增长率的新数据框。
    
    参数：
        periods (list): 报告期列表。
        percentage_changes (list): 各期的同比增长率（百分比变化）。
    
    返回：
        pd.DataFrame: 包含 'date' 和 '同比增长 (%)' 的数据框。
    """
    return pd.DataFrame({'date': periods, '同比增长 (%)': percentage_changes}).dropna()

if __name__ == "__main__":
    # 加载 Excel 文件并跳过前两列，保留数据从第三列开始
    df = pd.read_excel('全部A股 -1 .xlsx')
    df = df.iloc[:, 2:]

    # 将重命名逻辑应用于数据框的列名
    new_columns = [rename_column(col) for col in df.columns]

    # 将新列名应用到数据框
    df.columns = new_columns

    # 从第三列开始重命名列名（跳过 '证券代码' 和 '证券名称'）
    new_columns = [convert_to_date(col) for col in df.columns]
    new_columns = pd.to_datetime(new_columns, errors='coerce')  # 转换为日期格式，忽略错误
    df.columns = new_columns  # 应用新的列名

    # 将以文本存储的所有数字转换为数值类型（如有必要）
    df = df.apply(pd.to_numeric, errors='coerce')  # 遇到无法转换的值填充为 NaN


    # 计算同比增长率
    percentage_changes = calculate_percentage_change(df)

    # 从第五列开始的报告期
    periods = df.columns[4:]
    result_df = create_result_df(periods, percentage_changes)

    # 输出结果 DataFrame
    result_df

    # 设置中文字体以支持显示中文字符
    plt.rcParams['font.family'] = 'Arial Unicode MS'

    # 将前面计算结果存储的 dataframe 赋值为 final_df
    final_df = result_df
    final_df.to_csv('final_df.csv')  # Save as CSV

    # 加载第一个数据集（上证指数的历史数据）
    df1 = pd.read_csv('上证综合指数_20240911_134643.csv')  # 替换为实际文件路径或数据导入方式

    # 确保 'Date' 列为日期格式，并设置为索引
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.set_index('Date', inplace=True)

    # 去除 'Value' 列中的逗号，并将其转换为数值类型
    df1['Value'] = df1['Value'].replace({',': ''}, regex=True).astype(float)

    # 加载第二个数据集（财务数据结果）
    df2 = final_df.copy()  # 使用前面生成的最终数据框

    # 确保 'date' 列为日期格式，并设置为索引
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)

    # 筛选两个数据集的日期范围：2005-03-31 到 2024-06-30
    start_date = '2005-03-31'
    end_date = '2024-06-30'

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    # 使用 df2 的季度索引重新索引 df1，并使用前向填充的方法对数据进行对齐
    df1_quarterly = df1.reindex(df2.index, method='ffill')

    # 在同一张图中绘制两个数据集
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 在左侧 y 轴绘制上证指数
    ax1.plot(df1_quarterly.index, df1_quarterly['Value'], color='blue', label='上证指数 (左轴)')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('上证指数', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 在右侧 y 轴绘制归母净利润同比增长
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['同比增长 (%)'], color='orange', label='归母净利润同比增长 (右轴)')
    ax2.set_ylabel('上证指数与归母净利润同比 (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # 添加标题并显示图表
    plt.title('上证指数与归母净利润同比增速 (2005-03-31 到 2024-06-30)')

    # 设置 x 轴为季度日期显示格式
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 旋转 x 轴标签以提高可读性
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    plt.show()

    # 计算百分比变化（pct_change）
    df1_quarterly['pct_change'] = df1_quarterly['Value'].pct_change()
    df2['pct_change'] = df2['同比增长 (%)'].pct_change()

    # 删除由于 pct_change 计算而产生的 NaN 值
    df1_quarterly.dropna(subset=['pct_change'], inplace=True)
    df2.dropna(subset=['pct_change'], inplace=True)

    # 确保经过筛选和百分比变化计算后索引一致
    df1_quarterly = df1_quarterly.loc[df2.index]

    # 计算滚动十季度的相关性
    rolling_corr = df1_quarterly['pct_change'].rolling(window=10).corr(df2['pct_change'])

    # 绘制滚动相关性图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df1_quarterly.index, rolling_corr, color='blue', marker='o', label='变化相关性 (指数与净利润增长率)')
    ax.set_xlabel('日期')
    ax.set_ylabel('滚动十季度变化相关性')
    ax.set_ylim(-1, 1)
    ax.set_title('上证指数变化与净利润同比变化相关性')

    # 添加 y=0 的参考线
    ax.axhline(0, color='black', linestyle='--')

    # 设置 x 轴为季度日期显示格式
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 旋转 x 轴标签以提高可读性
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # 将两个数据的百分比变化合并到一个数据框中以进行 VAR 分析
    data = pd.concat([df1_quarterly['pct_change'], df2['pct_change']], axis=1)
    data.columns = ['Index Change', 'Profit Growth Change']

    # 绘制 CCF（交叉相关函数）
    plt.figure(figsize=(10, 6))
    ccf_values = ccf(data['Index Change'], data['Profit Growth Change'])
    plt.bar(range(len(ccf_values)), ccf_values)
    plt.title('交叉相关函数 (CCF)')
    plt.xlabel('滞后期')
    plt.ylabel('CCF 值')
    plt.tight_layout()
    plt.show()

    # 拟合 VAR 模型
    model = VAR(data)

    # 基于 AIC 和 BIC 选择最优滞后期
    lag_order_results = model.select_order(maxlags=10)
    print("滞后期选择结果:")
    print(lag_order_results.summary())

    # 根据 AIC/BIC 选择最优滞后期并确保至少为 1
    optimal_lag = max(int(abs(lag_order_results.aic)), 1)

    # 检查数据点是否足够支持选择的滞后期
    if len(data) <= optimal_lag:
        raise ValueError(f"数据点不足以支持选择的滞后期 ({optimal_lag})。请调整滞后期或添加更多数据。")

    var_model = model.fit(optimal_lag)

    # 打印模型的 AIC 和 BIC 值
    print(f"AIC: {var_model.aic}")
    print(f"BIC: {var_model.bic}")

    # 将原始数据与拟合值对齐
    fitted_values = var_model.fittedvalues
    aligned_index = data.index[optimal_lag:]

    # 绘制原始数据和拟合值的比较图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制指数变化的原始值和拟合值
    ax.plot(aligned_index, data['Index Change'][optimal_lag:], label='原始指数变化', color='blue')
    ax.plot(aligned_index, fitted_values['Index Change'], label='拟合指数变化', color='orange', linestyle='--')

    # 绘制利润增长的原始值和拟合值
    ax.plot(aligned_index, data['Profit Growth Change'][optimal_lag:], label='原始利润增长', color='green')
    ax.plot(aligned_index, fitted_values['Profit Growth Change'], label='拟合利润增长', color='red', linestyle='--')

    plt.title('原始数据与拟合数据对比 (指数变化与利润增长)')
    plt.xlabel('日期')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 计算脉冲响应函数（IRF）
    try:
        irf = var_model.irf(10)  # 设置脉冲响应周期为 10（可调整）
        irf.plot(orth=False)
        plt.show()
    except Exception as e:
        print(f"生成 IRF 时出错: {e}")

    # 格兰杰因果检验
    print(f'\n格兰杰因果检验: 指数变化 -> 利润增长')
    granger_test_1 = grangercausalitytests(data[['Index Change', 'Profit Growth Change']], maxlag=optimal_lag)
    print(f'结果: 上证指数变化 -> 归母净利润同比增长')
    print(granger_test_1)

    print(f'\n格兰杰因果检验: 利润增长 -> 指数变化')
    granger_test_2 = grangercausalitytests(data[['Profit Growth Change', 'Index Change']], maxlag=optimal_lag)
    print(f'结果: 归母净利润同比增长 -> 上证指数变化')
    print(granger_test_2)

