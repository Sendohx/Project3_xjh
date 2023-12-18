# -*- coding = utf-8 -*-
# @Time: 2023/12/18 16:52
# @Author: Jiahao Xu
# @File：CTABackTester_new.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages


class CTABacktester:
    """CTA回测基础框架"""

    def __init__(self, path, start_date, end_date, assets, weights, lower_limit, upper_limit, risk_free_rate=0.02,
                 min_acceptable_return=0.0):
        """
        : param start_date: 回测开始日期
        : param end_date: 回测结束日期
        : param assets: 标的
        : param weights: 标的初始权重
        : param lower_limit: 仓位下限
        : param upper_limit: 仓位上限
        : param risk_free_rate: 无风险收益率
        : param min_acceptable_return: 最小可接受回报率，用于计算索提诺比率
        """
        self.path = path
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets
        self.weights = weights
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.risk_free_rate = risk_free_rate
        self.min_acceptable_return = min_acceptable_return
        self.benchmark_returns = None
        self.strategy_returns = None
        self.excess_returns = None

    def get_benchmark_data(self):
        """获取基准数据"""
        # 这里假设基准数据保存在名为benchmark.csv的CSV文件中，包含日期和收益率列
        benchmark_data = pd.read_parquet('./data/benchmark.parquet')
        # benchmark_data['date'] = pd.to_datetime(benchmark_data['date'], format='%Y%m%d')

        # 根据回测起止时间筛选数据
        benchmark_data = benchmark_data.loc[
            (benchmark_data['date'] >= self.start_date) & (benchmark_data['date'] <= self.end_date)]
        benchmark_data.set_index('date', inplace=True)

        return benchmark_data

    def get_asset_data(self, asset):
        """
        获取单个标的数据
        :param asset: 标的名称
        """
        # 这里假设每个标的数据保存在以标的名称命名的CSV文件中，包含日期和收益率列
        asset_data = pd.read_parquet('./data/' + f'{asset}.parquet')
        # asset_data['date'] = pd.to_datetime(asset_data['date'], format='%Y%m%d')

        # 根据回测起止时间筛选数据
        asset_data = asset_data.loc[(asset_data['date'] >= self.start_date) & (asset_data['date'] <= self.end_date)]
        asset_data.set_index('date', inplace=True)

        return asset_data

    def get_portfolio_returns(self):  # 这个函数是不同策略的主要区别所在
        """获取组合收益率"""
        portfolio_returns = None

        for asset, weight in zip(self.assets, self.weights):
            asset_data = self.get_asset_data(asset)
            # return_data = asset_data['weighted_return']
            # return_data = return_data.rename(columns={'weighted_return':'return'})

            if portfolio_returns is None:
                portfolio_returns = asset_data['return'] * asset_data['weight']
            else:
                portfolio_returns += asset_data['return'] * asset_data['weight']

        portfolio_returns.rename('return')

        return portfolio_returns

    def execute_trades(self):
        """执行交易并计算策略收益率、基准收益率和超额收益率"""
        self.strategy_returns = self.get_portfolio_returns()
        self.benchmark_returns = self.get_benchmark_data()['return']
        self.excess_returns = ((self.strategy_returns + 1).cumprod() - (self.benchmark_returns + 1).cumprod()).diff()
        self.strategy_returns.iloc[0] = 0
        self.benchmark_returns.iloc[0] = 0
        self.excess_returns.iloc[0] = 0

    def plot_performance(self):
        """绘制策略和基准收益走势图"""
        fig1 = plt.figure(figsize=(25, 10))
        plt.rcParams['font.size'] = 16
        plt.rcParams["figure.autolayout"] = True
        plt.plot((self.strategy_returns + 1).cumprod() - 1, label='Strategy Returns')
        plt.plot((self.benchmark_returns + 1).cumprod() - 1, label='Benchmark Returns')
        plt.plot((self.strategy_returns + 1).cumprod() - (self.benchmark_returns + 1).cumprod(), label='Excess Returns')
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))
        plt.title('Return Trend')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Cumulative \n Returns',rotation=0, labelpad=40)
        # plt.yticks(rotation=45)

        return fig1

    def calculate_annual_return(self, returns):
        """
        计算年化收益
        :param returns: 收益序列
        """
        return ((returns + 1).prod()) ** (242 / len(returns)) - 1

    def calculate_annual_volatility(self, returns):
        """
        计算年化波动率
        :param returns: 收益序列
        """
        return returns.std() * np.sqrt(242)

    def calculate_max_drawdown(self, returns):
        """
        计算最大回撤
        :param returns: 收益序列
        """
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = peak - cum_returns
        max_drawdown = drawdown.max()
        return max_drawdown * 100

    def calculate_sharpe_ratio(self, returns):
        """
        计算夏普比率
        :param returns: 收益序列
        """
        return (self.calculate_annual_return(returns) - self.risk_free_rate) / self.calculate_annual_volatility(returns)

    def calculate_sortino_ratio(self, returns):
        """
        计算索提诺比率
        :param returns: 收益序列
        """
        downside_volatility = returns[returns < self.min_acceptable_return].std() * np.sqrt(242)
        return (self.calculate_annual_return(returns) - self.risk_free_rate) / downside_volatility

    def get_dataframe(self):
        """计算策略的各项指标"""
        metrics = dict()

        metrics['Annual Returns'] = [
            self.calculate_annual_return(self.benchmark_returns),
            self.calculate_annual_return(self.strategy_returns),
            self.calculate_annual_return(self.benchmark_returns) - self.calculate_annual_return(self.strategy_returns)]

        metrics['Annual Volatility'] = [
            self.calculate_annual_volatility(self.benchmark_returns),
            self.calculate_annual_volatility(self.strategy_returns),
            self.calculate_annual_volatility(self.excess_returns)]

        metrics['Max Drawdown'] = [
            self.calculate_max_drawdown(self.benchmark_returns),
            self.calculate_max_drawdown(self.strategy_returns),
            (self.excess_returns.cumsum().cummax() - self.excess_returns.cumsum()).max() * 100]

        metrics['Sharpe'] = [
            self.calculate_sharpe_ratio(self.benchmark_returns),
            self.calculate_sharpe_ratio(self.strategy_returns),
            (self.calculate_annual_return(self.benchmark_returns) - self.calculate_annual_return(
                self.strategy_returns) - self.risk_free_rate)
            / self.calculate_annual_volatility(self.excess_returns)]

        metrics['Sortino'] = [
            self.calculate_sortino_ratio(self.benchmark_returns),
            self.calculate_sortino_ratio(self.strategy_returns),
            (self.calculate_annual_return(self.benchmark_returns) - self.calculate_annual_return(
                self.strategy_returns) - self.risk_free_rate)
            / (self.excess_returns[self.excess_returns < self.min_acceptable_return].std() * np.sqrt(242))]

        df = pd.DataFrame(metrics)
        df = df.round(4)
        df.set_index([['benchmark', 'strategy', 'excess']], inplace=True)
        df.index.name = 'Backtest'
        df.reset_index(inplace=True)

        return df

    def plot_returns_distribution(self):
        """绘制收益分布直方图和统计指标"""
        fig2 = plt.figure(figsize=(25, 6))
        plt.rcParams['font.size'] = 16
        plt.rcParams["figure.autolayout"] = True

        # 基准收益分布
        plt.subplot(1, 3, 1)
        plt.hist(self.benchmark_returns, bins=30, alpha=0.5, label='Benchmark Returns')
        plt.axvline(self.benchmark_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.benchmark_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # 基准收益概率密度曲线
        plt.twinx()
        kde = gaussian_kde(self.benchmark_returns.values)
        x1 = np.linspace(np.min(self.benchmark_returns.values), np.max(self.benchmark_returns.values), 100)
        y1 = kde(x1)
        plt.plot(x1, y1, color='blue', label='KDE')
        plt.title('Benchmark Returns Distribution')
        plt.legend()

        # 策略收益分布
        plt.subplot(1, 3, 2)
        plt.hist(self.strategy_returns, bins=30, alpha=0.5, label='Strategy Returns')
        plt.axvline(self.strategy_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.strategy_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # 策略收益概率密度曲线
        plt.twinx()
        kde = gaussian_kde(self.strategy_returns.values)
        x2 = np.linspace(np.min(self.strategy_returns.values), np.max(self.strategy_returns.values), 100)
        y2 = kde(x2)
        plt.plot(x2, y2, color='blue', label='KDE')
        plt.title('Strategy Returns Distribution')
        plt.legend()

        # 超额收益分布
        plt.subplot(1, 3, 3)
        plt.hist(self.excess_returns, bins=30, alpha=0.5, label='Strategy Returns')
        plt.axvline(self.excess_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.excess_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # 超额收益概率密度曲线
        plt.twinx()
        kde = gaussian_kde(self.excess_returns.values)
        x3 = np.linspace(np.min(self.excess_returns.values), np.max(self.excess_returns.values), 100)
        y3 = kde(x3)
        plt.plot(x3, y3, color='blue', label='KDE')
        plt.title('Excess Returns Distribution')
        plt.legend()

        return fig2

    def return_stats(self):
        """
        计算收益的描述统计信息
        :param self:
        :return df: 包含均值、标准差、中位数、峰度和偏度的dataframe
        """
        values1 = self.benchmark_returns.values
        values2 = self.strategy_returns.values
        values3 = self.excess_returns.values
        values_list = [values1, values2, values3]

        df = pd.DataFrame()
        for value in values_list:
            stats = dict()

            stats['mean'] = np.round(np.mean(value), decimals=4)
            stats['std'] = np.round(np.std(value), decimals=4)
            stats['median'] = np.round(np.median(value), decimals=4)
            stats['kurtosis'] = np.round(kurtosis(value), decimals=4)
            stats['skewness'] = np.round(skew(value), decimals=4)

            df1 = pd.DataFrame(stats, index=[0])
            df = df._append(df1)

        df.set_index([['benchmark', 'strategy', 'excess']], drop=True, inplace=True)
        df.index.name = 'Stats'
        df.reset_index(inplace=True)

        return df

    def run_backtest(self):
        """运行回测框架,生成pdf"""
        self.execute_trades()
        with PdfPages(self.path) as pdf:
            pdf.savefig(self.plot_performance())

            df1 = self.get_dataframe()
            fig2, ax2 = plt.subplots(figsize=(25, 5))
            ax2.axis('off')  # Hide the axis
            table = ax2.table(cellText=df1.values, colLabels=df1.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(True)
            table.scale(1.5, 1.5)
            pdf.savefig(fig2)

            pdf.savefig(self.plot_returns_distribution())

            df2 = self.return_stats()
            fig4, ax4 = plt.subplots(figsize=(25,5))
            ax4.axis('off')  # Hide the axis
            table = ax4.table(cellText=df2.values, colLabels=df2.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(True)
            table.scale(1.5, 1.5)
            pdf.savefig(fig4)


if __name__ == '__main__':
    # 回测起止时间
    start_date = '20140101'
    end_date = '20231208'
    path = f'C:/Users/Sendoh/PycharmProjects/Project3_xjh/results/{start_date}_{end_date}.pdf'
    # 交易标的和对应权重
    assets = ['000985CSI']
    weights = [0.5]
    lower_limit = 0.0
    upper_limit = 1.0

    # 创建回测对象并执行回测
    backtester = CTABacktester(path, start_date, end_date, assets, weights, lower_limit, upper_limit)
    backtester.run_backtest()
