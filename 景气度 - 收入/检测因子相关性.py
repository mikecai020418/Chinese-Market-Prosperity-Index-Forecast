import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, ccf
import numpy as np

final_df = pd.read_csv('final_df.csv')
# 定义清理列的函数（移除逗号并转为数值）
def clean_column(col):
    """
    清理列数据，去除逗号并转换为数值类型。

    参数:
    - col (pandas.Series): 需要清理的列。

    返回:
    - pandas.Series: 清理后的列。
    """
    if col.dtype == 'object':  # 如果列是字符串，替换逗号
        return pd.to_numeric(col.str.replace(',', ''), errors='coerce')
    else:  # 如果已是数值，则直接转为数值型（处理NaN或非数值）
        return pd.to_numeric(col)

def analyze_data(file_name):
    """
    分析输入文件中的数据，执行多种数据清理、交叉相关分析和模型拟合操作。

    参数:
    - file_name (str): 文件名，指向包含数据的Excel文件。

    输出:
    - None（直接生成图表并打印结果）
    """
    # 加载第一个数据框 (CRB数据)
    df1 = pd.read_excel(file_name)
    df1['Date'] = pd.to_datetime(df1['Date'])
    invalid_dates = df1['Date'].isna().sum()  # 统计无效日期条目数
    print(f"无效日期条目数: {invalid_dates}")

    df1.set_index('Date', inplace=True)

    # 对df1的每列应用清理函数
    df1 = df1.apply(clean_column)
    
    # 加载第二个数据框（营业总总营业收入同比增长） - 假设已预加载为final_df
    df2 = final_df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)

    # 筛选两个数据框中的指定日期范围数据
    start_date = '2005-03-31'
    end_date = '2024-06-30'

    # 确保Date索引按时间顺序排序后再切片
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]
    
    # 遍历df1中的列
    for column in df1.columns:
        # 清理并对齐 '累计同比' 列
        if '累计同比' in column:
            # 将df1重新索引为df2的季度日期，向前填充
            df1_clean = df1[[column]].dropna()

            # 获取df1_clean中数据有效的最早日期
            earliest_valid_date = df1_clean.index.min()

            # 截取df2，仅保留最早有效日期之后的数据
            df2_truncated = df2[df2.index >= earliest_valid_date]

            # 将df1_clean向前填充，以匹配df2的日期
            df1_filled = df1_clean.reindex(df2_truncated.index, method='ffill')

            # 对齐数据: 仅保留df1_filled和df2都有值的日期
            combined_df_plot = pd.concat([df1_filled[column], df2_truncated['同比增长 (%)']], axis=1).dropna()

            # 用于分析的完整数据集
            combined_df_full = combined_df_plot.copy()
            
        # 处理非同比列（按季度重新索引并取均值）
        else:
            df1_clean = df1[[column]].dropna()

            # 找到df1_clean中的最早日期
            earliest_valid_date = df1_clean.index.min()
            df2_truncated = df2[df2.index >= earliest_valid_date]

            # 提取两个数据集的年份和季度
            df1_clean['quarter'] = df1_clean.index.to_period('Q')
            df2_truncated['quarter'] = df2_truncated.index.to_period('Q')

            # 按‘季度’分组并计算每季度的均值
            df1_quarterly_mean = df1_clean.groupby('quarter')[[column]].mean()

            # 第一步: 基于‘季度’合并
            combined_df_plot = pd.merge(
                df2_truncated[['同比增长 (%)', 'quarter']],  
                df1_quarterly_mean,  
                on='quarter',
                how='inner'
            ).dropna()

            # 第二步: 使用df2_truncated的索引
            combined_df_plot.index = df2_truncated.index[:len(combined_df_plot)]  

            # 第三步: 删除‘季度’列
            combined_df_plot.drop(columns='quarter', inplace=True)

            # 第四步: 保存完整数据集用于分析
            combined_df_full = combined_df_plot.copy()

        # 绘图
        if len(combined_df_plot) > 0:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # 绘制df1的列数据
            ax1.plot(combined_df_plot.index, combined_df_plot[column], color='orange', label=f'{column}')
            ax1.set_xlabel('日期')
            ax1.set_ylabel(column, color='orange')
            ax1.tick_params(axis='y', labelcolor='orange')

            # 在第二个Y轴绘制营业总总营业收入同比增长
            ax2 = ax1.twinx()
            ax2.plot(combined_df_plot.index, combined_df_plot['同比增长 (%)'], color='blue', label='营业总总营业收入同比增长')
            ax2.set_ylabel('营业总总营业收入同比 (%)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            plt.title(f'{column} 与 营业总总营业收入同比增长 的对比')
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=7))  
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()
            plt.show()

        # 计算交叉相关函数（CCF）
        cross_corr = ccf(combined_df_full[column], combined_df_full['同比增长 (%)'])
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(cross_corr)), cross_corr)
        plt.title(f'交叉相关: {column} 和 营业总总营业收入同比增长')
        plt.show()

        # 计算10个季度的滚动相关性
        rolling_corr = combined_df_full[column].rolling(window=10).corr(combined_df_full['同比增长 (%)'])
        plt.figure(figsize=(10, 6))
        plt.plot(rolling_corr.index, rolling_corr, color='green', label=f'10季度滚动相关性')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'10季度滚动相关性: {column} 和 营业总总营业收入同比增长')
        plt.legend()
        plt.show()

        # VAR模型拟合
        model = VAR(combined_df_full)
        aic_bic = {}
        for lag in range(1, 6):
            result = model.fit(lag)
            aic_bic[lag] = {'AIC': result.aic, 'BIC': result.bic}

        # 选择最佳滞后期
        best_lag = min(aic_bic, key=lambda x: (aic_bic[x]['AIC'], aic_bic[x]['BIC']))
        best_lag = max(int(abs(best_lag)), 1)
        print(f'基于AIC/BIC的最佳滞后期: {best_lag}')
        
        # 检查是否有足够数据点
        if len(combined_df_full) <= best_lag:
            raise ValueError(f"数据点不足以进行滞后期 {best_lag} 分析")

        best_model = model.fit(best_lag)
        fitted_values = best_model.fittedvalues

        # 绘制原始值与拟合值对比
        plt.figure(figsize=(10, 6))
        plt.plot(combined_df_full.index, combined_df_full[column], color='orange', label=f'原始 {column}')
        plt.plot(fitted_values.index, fitted_values[column], color='red', linestyle='--', label=f'拟合 {column}')
        plt.plot(combined_df_full.index, combined_df_full['同比增长 (%)'], color='blue', label='原始 营业总总营业收入同比增长')
        plt.plot(fitted_values.index, fitted_values['同比增长 (%)'], color='purple', linestyle='--', label='拟合 营业总总营业收入同比增长')
        plt.title(f'原始值 vs 拟合值: {column} 和 营业总总营业收入同比增长')
        plt.legend()
        plt.show()

        # 冲击响应函数（IRF）
        try:
            irf = best_model.irf(10)
            irf.plot(orth=False)
            plt.show()
        except Exception as e:
            print(f"生成IRF时出错: {e}")

        # Granger因果检验
        print(f'\nGranger因果检验: 收入 -> {column}')
        granger_test_1 = grangercausalitytests(combined_df_full[['同比增长 (%)', column]], maxlag=best_lag)
        print(f'检验结果: 营业总总营业收入同比增长 -> {column}')

        print(f'\nGranger因果检验: {column} -> 收入')
        granger_test_2 = grangercausalitytests(combined_df_full[[column, '同比增长 (%)']], maxlag=best_lag)
        print(f'检验结果: {column} -> 营业总总营业收入同比增长')


def analyze_group(group_name, group_df, target_series, use_average=True):
    """
    分析特定组数据与目标时间序列之间的关系，包括绘图、交叉相关性、滚动相关性、向量自回归（VAR）建模及格兰杰因果检验。

    参数:
    - group_name (str): 数据组的名称。
    - group_df (DataFrame): 数据组的时间序列数据。
    - target_series (Series): 目标时间序列（例如“同比增长 (%)”）。
    - use_average (bool): 是否对数据组取均值。默认为True。

    返回:
    无返回值，函数将绘制相关图表并打印分析结果。
    """
    # 如果use_average为True，计算数据组的均值
    if use_average:
        group_series = group_df.mean(axis=1)  # 对数据组取均值
        group_series.name = f'{group_name} 平均值'  # 为系列赋予名称
    else:
        group_series = group_df.squeeze()  # 如果数据组只有一列，将其转换为Series
        if group_series.name is None:
            group_series.name = group_name  # 如果Series没有名称，赋予组名

    # 数据对齐：仅保留组数据和目标序列都有值的日期
    combined_df_plot = pd.concat([group_series, target_series], axis=1).dropna()
    group_series_aligned = combined_df_plot[group_series.name]
    target_series_aligned = combined_df_plot['同比增长 (%)']

    # 数据绘图
    if len(combined_df_plot) > 0:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制组平均值或单列数据
        ax1.plot(combined_df_plot.index, combined_df_plot[group_series.name], color='orange', label=f'{group_name}')
        ax1.set_xlabel('日期')
        ax1.set_ylabel(group_name, color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')

        # 创建右侧y轴，绘制营业收入同比增长数据
        ax2 = ax1.twinx()
        ax2.plot(combined_df_plot.index, combined_df_plot['同比增长 (%)'], color='blue', label='营业总收入同比增长 (右轴)')
        ax2.set_ylabel('营业总收入同比增长 (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title(f'{group_name} 与 营业总收入同比增长 的对比')

        # 设置x轴显示季度日期
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=7))  # 设置间隔为7个月
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化为 年-月
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        fig.tight_layout()
        plt.show()

    # 计算交叉相关性（CCF）
    cross_corr = ccf(group_series_aligned, target_series_aligned)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cross_corr)), cross_corr)
    plt.title(f'交叉相关性: {group_name} 和 营业总收入同比增长')
    plt.show()

    # 10季度滚动相关性
    rolling_corr = group_series_aligned.rolling(window=10).corr(target_series_aligned)

    # 绘制滚动相关性
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_corr.index, rolling_corr, label=f'{group_name} 和 营业总收入同比增长 的10季度滚动相关性', color='green')
    plt.title(f'10季度滚动相关性: {group_name} 和 营业总收入同比增长')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.show()

    # 向量自回归(VAR)模型拟合和选择
    combined_df_full = pd.concat([group_series, target_series], axis=1).dropna()
    model = VAR(combined_df_full)
    aic_bic = {}
    for lag in range(1, 6):
        result = model.fit(lag)
        aic_bic[lag] = {'AIC': result.aic, 'BIC': result.bic}

    # 根据AIC/BIC选择最佳滞后期
    best_lag = min(aic_bic, key=lambda x: (aic_bic[x]['AIC'], aic_bic[x]['BIC']))
    best_lag = max(int(abs(best_lag)), 1)
    print(f'{group_name} 的最佳滞后期 (基于AIC和BIC): {best_lag}')

    # 检查数据点是否足够支持所选滞后期
    if len(combined_df_full) <= best_lag:
        raise ValueError(f"数据点不足以支持滞后期 ({best_lag})。请调整滞后期或添加更多数据。")

    # 拟合最佳模型
    best_model = model.fit(best_lag)

    # 冲击响应函数 (IRF)
    try:
        irf = best_model.irf(10)
        irf.plot(orth=False)
        plt.show()
    except Exception as e:
        print(f"生成IRF时出错: {e}")

    # 格兰杰因果检验
    print(f'\n格兰杰因果检验: Profit -> {group_name}')
    granger_test_1 = grangercausalitytests(combined_df_full[['同比增长 (%)', group_series.name]], maxlag=best_lag)
    print(f'营业总收入同比增长 -> {group_name} 的检验结果:')
    print(granger_test_1)

    print(f'\n格兰杰因果检验: {group_name} -> Profit')
    granger_test_2 = grangercausalitytests(combined_df_full[[group_series.name, '同比增长 (%)']], maxlag=best_lag)
    print(f'{group_name} -> 营业总收入同比增长 的检验结果:')
    print(granger_test_2)

if __name__ == "__main__":
    # 做granger causality需要正态分布数据？
    # second order covariance stationary
    '''单项分析'''
    analyze_data('房地产开发投资完成额建筑工程累计同比等_20240914_143019.xlsx')
    analyze_data('税收收入关税累计同比等_20240918_155253.xlsx')
    analyze_data('家用电冰箱产量当月同比等_20240918_165840.xlsx')
    analyze_data('规模以上工业增加值国有控股企业当月同比等_20240919_125549.xlsx')


    '''中观因子合并分析'''
    df1 = pd.read_excel('家用电冰箱产量当月同比等_20240918_165840.xlsx')
    df1['Date'] = pd.to_datetime(df1['Date'])
    invalid_dates = df1['Date'].isna().sum()
    print(f"Number of invalid date entries: {invalid_dates}")

    # 将 'Date' 列设置为索引
    df1.set_index('Date', inplace=True)

    # 定义一个清洗列的函数（去掉逗号并转换为数值类型）
    def clean_column(col):
        if col.dtype == 'object':  # 如果列包含字符串，则替换逗号
            return pd.to_numeric(col.str.replace(',', ''), errors='coerce')
        else:  # 如果列已经是数值类型，直接转换以处理潜在的NaN或非数值值
            return pd.to_numeric(col)

    # 将清洗函数应用到 df1 的每一列
    df1 = df1.apply(clean_column)

    # 加载第二个数据框 (营业总总营业收入同比增长)
    df2 = final_df.copy()  # 替换为正确的文件路径或数据导入方法
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)

    # 按指定日期范围过滤两个数据框
    start_date = '2001-12-31'
    end_date = '2024-06-30'

    # 确保索引按时间顺序排序后再进行切片
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    # 定义分组策略
    group1_cols = df1.columns[:3]  # 前三列
    group2_cols = df1.columns[3:6]  # 接下来的三列
    individual_cols = df1.columns[6:8]  # 再接下来的两列（不需要求平均）
    last_col = df1.columns[8]  # 最后一列

    # 分析第 1 组数据（计算平均值）
    analyze_group("家电产量因子", df1[group1_cols], df2['同比增长 (%)'], use_average=True)

    # 分析第 2 组数据（计算平均值）
    analyze_group("机械产量因子", df1[group2_cols], df2['同比增长 (%)'], use_average=True)

    # -----------------------------------------------------------
    # 加载第一个数据框 (CRB 数据)
    df1 = pd.read_excel('家用电冰箱产量当月同比等_20240918_165840.xlsx')
    df1['Date'] = pd.to_datetime(df1['Date'])

    # 检查无效日期条目数量
    invalid_dates = df1['Date'].isna().sum()
    print(f"无效日期条目的数量: {invalid_dates}")

    # 将 'Date' 列设置为索引
    df1.set_index('Date', inplace=True)

    # 将清洗函数应用到 df1 的每一列
    df1 = df1.apply(clean_column)

    # 加载第二个数据框 (营业总总营业收入同比增长)
    df2 = final_df.copy()  # 替换为正确的文件路径或数据导入方法
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)

    # 按指定日期范围过滤两个数据框
    start_date = '2001-12-31'
    end_date = '2024-06-30'

    # 确保索引按时间顺序排序后再进行切片
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    # 重建 df1，使其与 df2 的季度日期对齐，并用前值填充缺失值
    df1_cleaned = pd.DataFrame()

    for column in df1.columns:
        # 清洗当前列（删除 NaN 值）
        df1_clean = df1[[column]].dropna()

        # 找到 df1_clean 中最早的数据日期
        earliest_valid_date = df1_clean.index.min()

        # 截取 df2，仅保留从最早有效日期开始的记录
        df2_truncated = df2[df2.index >= earliest_valid_date]

        # 将 df1_clean 重建为 df2_truncated 的季度日期索引，并用前值填充
        df1_quarterly = df1_clean.reindex(df2_truncated.index, method='ffill')

        # 将清洗后的季度数据合并到一个新数据框
        df1_cleaned = pd.concat([df1_cleaned, df1_quarterly], axis=1)

    # 以下绘制三个图表，并对齐 y 轴
    # 图表 1：前三列数据
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[0]], label='产量: 家用电冰箱: 当月同比 (%)', color='green')
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[1]], label='产量: 家用洗衣机: 当月同比 (%)', color='brown')
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[2]], label='产量: 空调: 当月同比 (%)', color='red')

    # 自定义 x 轴和标签
    ax1.set_xlabel('日期')
    ax1.set_ylabel('产量同比增长 (%)')

    # 左 y 轴设置适当的范围以对齐数据
    ax1.set_ylim(-60, 120)

    # 添加右 y 轴以显示 df2 的最后一列
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['同比增长 (%)'], label='上证指数总营业收入同比 (%)', color='blue', linewidth=2)
    ax2.set_ylabel('上证指数总营业收入同比 (%)', color='blue')

    # 设置右 y 轴范围
    ax2.set_ylim(-50, 150)

    # 设置 x 轴显示季度日期
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 添加标题和图例
    plt.title('家电产量因子与上证指数总营业收入同比增长 (第一部分)')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # 显示图表
    plt.tight_layout()
    plt.show()

    # 绘图 2：接下来的三列数据
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制前三列的数据曲线
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[3]], label=f'{df1_cleaned.columns[3]}', color='green')
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[4]], label=f'{df1_cleaned.columns[4]}', color='brown')
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[5]], label=f'{df1_cleaned.columns[5]}', color='red')

    # 自定义 x 轴和标签
    ax1.set_xlabel('日期')
    ax1.set_ylabel('产量同比增长 (%)')

    # 设置左 y 轴范围以便与 df2 的趋势对齐
    ax1.set_ylim(-100, 300)  # 设置第二个图表的轴范围

    # 添加右侧 y 轴用于显示 df2 的最后一列
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['同比增长 (%)'], label='上证指数总营业收入同比 (%)', color='blue', linewidth=2)
    ax2.set_ylabel('上证指数总营业收入同比 (%)', color='blue')

    # 设置右侧 y 轴范围
    ax2.set_ylim(-50, 150)  # 设置右侧 y 轴的范围

    # 设置 x 轴显示季度日期
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # 调整时间间隔
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 添加标题和图例
    plt.title('机械产量因子与上证指数总营业收入同比增长 (第二部分)')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # 显示图表
    plt.tight_layout()
    plt.show()

    # 绘图 3：最后两列数据 (包括右侧 y 轴用于显示百分比)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制第一列数据
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[6]], label=f'{df1_cleaned.columns[6]}', color='green')

    # 自定义 x 轴和标签
    ax1.set_xlabel('日期')
    ax1.set_ylabel('企业景气指数 (%)')

    # 调整左侧 y 轴范围以便正确对齐
    ax1.set_ylim(50, 210)  # 设置企业景气指数的范围

    # 添加右侧 y 轴用于同时显示 df2 的最后一列和 df1_cleaned.columns[7]
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['同比增长 (%)'], label='上证指数总营业收入同比 (%)', color='blue', linewidth=2)
    ax2.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[7]], label=f'{df1_cleaned.columns[7]}', color='red', linestyle='dashed')

    # 为右侧 y 轴设置范围和标签
    ax2.set_ylabel('上证指数总营业收入同比 (%) 和 百分比', color='blue')
    ax2.set_ylim(-40, 120)  # 设置右侧 y 轴的范围

    # 设置 x 轴显示季度日期
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # 调整时间间隔
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 添加标题和图例
    plt.title('钢铁产量因子与上证指数总营业收入同比增长 (第三部分)')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # 显示图表
    plt.tight_layout()
    plt.show()

    '''最后图表'''

    df1 = pd.read_excel('规模以上工业增加值国有控股企业当月同比等_20240919_125549.xlsx')
    df1['Date'] = pd.to_datetime(df1['Date'])
    # 检查是否有无效的日期条目
    invalid_dates = df1['Date'].isna().sum()
    print(f"无效日期条目的数量: {invalid_dates}")

    # 将日期设置为索引
    df1.set_index('Date', inplace=True)

    # 清理列的函数 (去除逗号，转换为数值型)
    def clean_column(col):
        if col.dtype == 'object':  # 如果列包含字符串，替换逗号
            return pd.to_numeric(col.str.replace(',', ''), errors='coerce')
        else:  # 如果已经是数值型，转换为数值型 (以处理可能的 NaN 或非数值值)
            return pd.to_numeric(col)

    # 对 df1 的每一列应用清理函数
    df1 = df1.apply(clean_column)

    # 加载第二个数据框 (营业总总营业收入同比增长)
    df2 = final_df.copy()  # 替换为正确的文件路径或数据导入方法
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)

    # 筛选两个数据框中指定日期范围内的数据
    start_date = '2001-12-31'
    end_date = '2024-06-30'

    # 确保日期索引按时间排序
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    # 将 df1 重新索引为 df2 的季度日期，并向前填充缺失值
    df1_cleaned = pd.DataFrame()

    for column in df1.columns:
        # 清理指定列，删除 NaN 值
        df1_clean = df1[[column]].dropna()

        # 找到 df1_clean 中数据可用的最早日期
        earliest_valid_date = df1_clean.index.min()

        # 截断 df2 以仅包括该最早有效日期及之后的日期
        df2_truncated = df2[df2.index >= earliest_valid_date]

        # 将 df1_clean 重新索引为 df2_truncated 的季度日期，并向前填充
        df1_quarterly = df1_clean.reindex(df2_truncated.index, method='ffill')

        # 将清理后的季度数据合并到新数据框中
        df1_cleaned = pd.concat([df1_cleaned, df1_quarterly], axis=1)

    # 绘图 1：前三列
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 df1_cleaned 第一列数据曲线
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[0]], label=f'{df1_cleaned.columns[0]} (% 左轴)', color='green')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('产量同比增长 (%)')
    ax1.set_ylim(-3, 20)  # 调整左 y 轴范围

    # 添加右 y 轴用于显示 df2 和 df1_cleaned 的第二列和第三列
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['同比增长 (%)'], label='上证指数总营业收入同比 (% 右轴)', color='blue', linewidth=2)
    ax2.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[1]], label=f'{df1_cleaned.columns[1]} (% 右轴)', color='brown')
    ax2.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[2]], label=f'{df1_cleaned.columns[2]} (% 右轴)', color='red')
    ax2.set_ylabel('上证指数总营业收入同比 (%)', color='blue')
    ax2.set_ylim(-65, 200)  # 调整右 y 轴范围

    # 设置 x 轴显示季度日期
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 添加标题
    plt.title('工业利润因子与上证指数总营业收入同比增长 (第一部分)')

    # 调整图例大小并将其放置在图内
    fig.legend(loc='upper right', bbox_to_anchor=(0.93, 0.95), fontsize='medium')

    # 显示图表
    plt.tight_layout()
    plt.show()

    # 绘图 2：接下来的两列
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 df1_cleaned 第四列数据曲线
    ax1.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[3]], label=f'{df1_cleaned.columns[3]} (左轴)', color='green')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('产量同比增长 (%)')
    ax1.set_ylim(0, 12000)  # 调整左 y 轴范围

    # 添加右 y 轴用于显示 df2 和 df1_cleaned 的第五列
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['同比增长 (%)'], label='上证指数总营业收入同比 (% 右轴)', color='blue', linewidth=2)
    ax2.plot(df1_cleaned.index, df1_cleaned[df1_cleaned.columns[4]], label=f'{df1_cleaned.columns[4]} (% 右轴)', color='brown')
    ax2.set_ylabel('上证指数总营业收入同比 (%)', color='blue')
    ax2.set_ylim(-40, 120)  # 调整右 y 轴范围

    # 设置 x 轴显示季度日期
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 添加标题
    plt.title('经济活跃度因子与上证指数总营业收入同比增长 (第二部分)')

    # 调整图例大小并将其放置在图内
    fig.legend(loc='upper right', bbox_to_anchor=(0.93, 0.95), fontsize='medium')

    # 显示图表
    plt.tight_layout()
    plt.show()




