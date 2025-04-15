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
# update data to newest? 
def rename_column(col_name):
    """  
    从列名中提取报告期并返回结果。  

    该函数旨在处理特定格式的列名，通过查找"[年度]"和"[报表类型]"的关键词，提取其中的报告期部分。
    如果列名格式不符合预期，将直接返回原列名。  

    参数：  
    col_name (str): 列名字符串，包含"[年度]"和"[报表类型]"等关键词。

    返回：  
    str: 提取后的报告期字符串。如果格式异常，返回原始列名。  
    """
    # 查找关键词 "[年度]" 的起始位置
    start = col_name.find("[年度]")
    # 查找关键词 "[报表类型]" 的起始位置
    end = col_name.find("[报表类型]")
    
    if start != -1 and end != -1 and end > start:
        # 提取 "[年度]" 与 "[报表类型]" 之间的字符串，并去除多余空格
        return col_name[start + len("[年度]"):end].strip()
    else:
        # 如果关键词位置不符合预期，返回原列名
        return col_name

def convert_to_date(col_name):
    """
    将报告期字符串转换为日期格式。

    参数:
        col_name (str): 报告期列名，例如 "2023年年报", "2023年中报"。

    返回:
        pd.Timestamp: 对应报告期的日期。
    """
    if '季报' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-03-31")  # 第一季度
    elif '中报' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-06-30")  # 第二季度
    elif '年报' in col_name:
        return pd.Timestamp(f"{col_name[:4]}-09-30")  # 第三季度
    else:
        return pd.Timestamp(f"{col_name[:4]}-12-31")  # 全年（第四季度）


# 主程序

''' 1. 计算2001-12-31以来收入 '''

df = pd.read_excel('沪深A股 - 收入.xlsx')  # 读取Excel文件
df = df.iloc[:, 2:]  # 删除前两列，保留数据列

# 步骤1：通过提取报告期（例如'1990一季', '1990中期'）重命名列名

# 将重命名逻辑应用到所有列
new_columns = [rename_column(col) for col in df.columns]

# 将新的列名赋值给DataFrame
df.columns = new_columns

# 从第三列开始重新命名列（跳过'证券代码'和'证券名称'）
new_columns = [convert_to_date(col) for col in df.columns]
new_columns = pd.to_datetime(new_columns)  # 确保列名为日期格式
# 将新的列名赋值给DataFrame
df.columns = new_columns

# 步骤2：将存储为文本的数字值转换为数值型（如有必要）
df = df.apply(pd.to_numeric, errors='coerce')  # 将无法转换的值替换为NaN

# 步骤3：计算每个报告期的净利润同比增长百分比
percentage_changes = []

# 从第五列开始遍历（确保可以进行同比比较）
for i in range(4, len(df.columns)):
    prev_col = df.columns[i-4]  # 获取一年前的列
    current_col = df.columns[i]  # 当前列
    
    # 筛选当前列和前一年列中都有有效数值的行
    valid_rows = df[[prev_col, current_col]].dropna()
    
    # 分别计算前一年和当前期的净利润总和
    prev_sum = valid_rows[prev_col].sum()
    current_sum = valid_rows[current_col].sum()
    
    # 计算净利润同比增长百分比
    percentage_change = ((current_sum - prev_sum) / prev_sum) * 100 if prev_sum != 0 else None
    percentage_changes.append(percentage_change)

# 步骤4：将结果导出为新的DataFrame
periods = df.columns[4:]  # 获取从第五列开始的报告期
result_df = pd.DataFrame({'date': periods, '同比增长 (%)': percentage_changes})

# 删除包含空值的行
result_df = result_df.dropna()

# 输出结果DataFrame（可选）
result_df

'''2. 检验收入与上证指数关系'''

final_df = result_df
# 加载第一个数据框 (股票价格)
df1 = pd.read_csv('上证综合指数_20240911_134643.csv')  # 替换为正确的文件路径或数据导入方法

# 确保 'Date' 列为日期格式，并将其设置为索引
df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)

# 去掉 'Value' 列中的逗号并转换为浮点数
df1['Value'] = df1['Value'].replace({',': ''}, regex=True).astype(float)

# 加载第二个数据框 (财务数据)
df2 = final_df.copy()  # 这是指之前步骤中创建的数据框

# 确保 'date' 列为日期格式，并将其设置为索引
df2['date'] = pd.to_datetime(df2['date'])
df2.set_index('date', inplace=True)

# 筛选两个数据框的时间范围为 2005-03-31 到 2024-06-30
start_date = '2005-03-31'
end_date = '2024-06-30'

df1 = df1.loc[start_date:end_date]
df2 = df2.loc[start_date:end_date]

# 使用 reindex 将 df1 调整为 df2 的季度日期，并向前填充股票价格数据
df1_quarterly = df1.reindex(df2.index, method='ffill')

# 在同一图表上绘制两个数据框的数据
fig, ax1 = plt.subplots(figsize=(10, 6))

# 在左侧坐标轴绘制股票价格
ax1.plot(df1_quarterly.index, df1_quarterly['Value'], color='blue', label='上证指数 (左轴)')
ax1.set_xlabel('日期')
ax1.set_ylabel('上证指数', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建右侧坐标轴用于显示财务增长数据 (营业总总营业收入同比增长)
ax2 = ax1.twinx()
ax2.plot(df2.index, df2['同比增长 (%)'], color='orange', label='营业总总营业收入同比增长 (右轴)')
ax2.set_ylabel('同比增长 (%)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 添加标题并显示图表
plt.title('上证指数与营业总总营业收入同比增速 (2005-03-31 到 2024-06-30)')

# 将 x 轴设置为按季度显示日期
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=7))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 旋转 x 轴标签以提高可读性
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

fig.tight_layout()
plt.show()

# 计算百分比变化
df1_quarterly['pct_change'] = df1_quarterly['Value'].pct_change()
df2['pct_change'] = df2['同比增长 (%)'].pct_change()

# 删除由于百分比变化计算导致的 NaN 行
df1_quarterly.dropna(subset=['pct_change'], inplace=True)
df2.dropna(subset=['pct_change'], inplace=True)

# 确保经过筛选和百分比变化计算后的索引匹配
df1_quarterly = df1_quarterly.loc[df2.index]

# 计算滚动 10 季度的相关性
rolling_corr = df1_quarterly['pct_change'].rolling(window=10).corr(df2['pct_change'])

# 绘制滚动相关性图
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df1_quarterly.index, rolling_corr, color='blue', marker='o', label='指数变化与净利润增长变化的相关性')
ax.set_xlabel('日期')
ax.set_ylabel('滚动十季度变化相关性')
ax.set_ylim(-1, 1)
ax.set_title('上证指数变化与营业总总营业收入同比变化相关性')

# 添加 y=0 的水平线
ax.axhline(0, color='black', linestyle='--')

# 设置 x 轴按季度显示日期
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 旋转 x 轴标签以提高可读性
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 将数据合并到一个数据框中用于 VAR 分析
data = pd.concat([df1_quarterly['pct_change'], df2['pct_change']], axis=1)
data.columns = ['指数变化', '利润增长变化']

# 绘制 CCF (互相关函数)
plt.figure(figsize=(10, 6))
ccf_values = ccf(data['指数变化'], data['利润增长变化'])
plt.bar(range(len(ccf_values)), ccf_values)
plt.title('互相关函数 (CCF)')
plt.xlabel('滞后')
plt.ylabel('CCF')
plt.tight_layout()
plt.show()

# 拟合 VAR 模型
model = VAR(data)

# 基于 AIC、BIC 选择最优滞后期
lag_order_results = model.select_order(maxlags=10)
print("滞后期选择结果:")
print(lag_order_results.summary())

# 使用最优滞后期拟合 VAR 模型
optimal_lag = max(int(abs(lag_order_results.aic)), 1)  # 确保滞后期至少为 1

# 检查数据点数量是否足够拟合选定滞后期的模型
if len(data) <= optimal_lag:
    raise ValueError(f"数据点不足以拟合选定的滞后期 ({optimal_lag})。请调整滞后期或增加数据。")

var_model = model.fit(optimal_lag)

# 输出模型的 AIC 和 BIC
print(f"AIC: {var_model.aic}")
print(f"BIC: {var_model.bic}")

# 绘制拟合值与原始数据的对比
fitted_values = var_model.fittedvalues

# 对齐拟合值与原始数据索引
aligned_index = data.index[optimal_lag:]

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制原始和拟合的指数变化
ax.plot(aligned_index, data['指数变化'][optimal_lag:], label='原始指数变化', color='blue')
ax.plot(aligned_index, fitted_values['指数变化'], label='拟合指数变化', color='orange', linestyle='--')

# 绘制原始和拟合的利润增长变化
ax.plot(aligned_index, data['利润增长变化'][optimal_lag:], label='原始利润增长', color='green')
ax.plot(aligned_index, fitted_values['利润增长变化'], label='拟合利润增长', color='red', linestyle='--')

plt.title('原始数据与拟合数据 (指数变化与利润增长变化)')
plt.xlabel('日期')
plt.legend()
plt.tight_layout()
plt.show()

# 冲击响应函数 (IRF)
try:
    irf = var_model.irf(10)  # 10 期冲击响应 (可根据需要调整期数)
    irf.plot(orth=False)
    plt.show()
except Exception as e:
    print(f"生成冲击响应函数时出错: {e}")

# 检验 1: 营业总总营业收入同比增长 是否 Granger 导致 上证指数增长
print(f'\nGranger 因果检验: 指数变化 -> 收入')
granger_test_1 = grangercausalitytests(data[['指数变化', '利润增长变化']], maxlag=optimal_lag)
print(f'上证指数增长 -> 营业总总营业收入同比增长 的结果:')
print(granger_test_1)

# 检验 2: 上证指数增长 是否 Granger 导致 营业总总营业收入同比增长
print(f'\nGranger 因果检验: 收入 -> 指数变化')
granger_test_2 = grangercausalitytests(data[['利润增长变化', '指数变化']], maxlag=optimal_lag)
print(f'营业总总营业收入同比增长 -> 上证指数增长 的结果:')
print(granger_test_2)


# 加一个word讲下每个文件是干啥的
