# -*- coding = utf-8 -*-
# @Time: 2023/12/08 15:52
# @Author: Jiahao Xu
# @File：CTABackTester.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, gaussian_kde


class CTABacktester:
    """CTA回测基础框架"""
    def __init__(self, start_date, end_date, assets, weights):
        """
        : param start_date: 回测开始日期
        : param end_date: 回测结束日期
        : param assets: 标的
        : param weights: 标的权重
        """
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets
        self.weights = weights
        self.benchmark_returns = None
        self.strategy_returns = None
        self.excess_returns = None

    def get_benchmark_data(self):
        """获取基准数据"""
        # 这里假设基准数据保存在名为benchmark.csv的CSV文件中，包含日期和收益率列
        benchmark_data = pd.read_csv('benchmark.csv', parse_dates=['date'])
        benchmark_data.set_index('date', inplace=True)

        # 根据回测起止时间筛选数据
        benchmark_data = benchmark_data.loc[self.start_date:self.end_date]

        return benchmark_data

    def get_asset_data(self, asset):
        """
        获取单个标的数据
        :param asset: 标的名称
        """
        # 这里假设每个标的数据保存在以标的名称命名的CSV文件中，包含日期和收益率列
        asset_data = pd.read_csv(f'{asset}.csv', parse_dates=['date'])
        asset_data.set_index('date', inplace=True)

        # 根据回测起止时间筛选数据
        asset_data = asset_data.loc[self.start_date: self.end_date]

        return asset_data

    def get_portfolio_returns(self):
        """获取组合收益率"""
        portfolio_returns = None

        for asset, weight in zip(self.assets, self.weights):
            asset_data = self.get_asset_data(asset)

            if portfolio_returns is None:
                portfolio_returns = asset_data['returns'] * weight
            else:
                portfolio_returns += asset_data['returns'] * weight

        return portfolio_returns

    def execute_trades(self):
        """执行交易并计算策略收益率、基准收益率和超额收益率"""
        self.strategy_returns = self.get_portfolio_returns()
        self.benchmark_returns = self.get_benchmark_data()['returns']
        self.excess_returns = self.strategy_returns - self.benchmark_returns

    def plot_performance(self):
        """绘制策略和基准收益走势图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.strategy_returns.cumsum(), label='Strategy Returns')
        plt.plot(self.benchmark_returns.cumsum(), label='Benchmark Returns')
        plt.plot(self.excess_returns.cumsum(), label='Excess Returns')
        plt.legend()
        plt.title('Return Trend')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()

    def calculate_max_drawdown(self, returns):
        """
        计算最大回撤
        :param returns: 收益序列
        """
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = cum_returns / peak - 1
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_sharpe_ratio(self, returns):
        """
        计算夏普比率
        :param returns: 收益序列
        """
        return (returns.mean() - 0.02) / returns.std()

    def calculate_sortino_ratio(self, returns):
        """
        计算索提诺比率
        :param returns: 收益序列
        """
        downside_returns = returns[returns < 0]
        return (returns.mean() - 0.02) / downside_returns.std()

    def get_dataframe(self):
        """计算策略的各项指标"""
        metrics = {}

        metrics['Annualized Returns'] = [
            self.benchmark_returns.mean() * 252,
            self.strategy_returns.mean() * 252,
            self.excess_returns.mean() * 252]

        metrics['Annualized Volatility'] = [
            self.benchmark_returns.std() * np.sqrt(252),
            self.strategy_returns.std() * np.sqrt(252),
            self.excess_returns.std() * np.sqrt(252)]

        metrics['Max Drawdown'] = [
            self.calculate_max_drawdown(self.benchmark_returns),
            self.calculate_max_drawdown(self.strategy_returns),
            self.calculate_max_drawdown(self.excess_returns)]

        metrics['Sharpe Ratio'] = [
            self.calculate_sharpe_ratio(self.benchmark_returns),
            self.calculate_sharpe_ratio(self.strategy_returns),
            self.calculate_sharpe_ratio(self.excess_returns)]

        metrics['Sortino Ratio'] = [
            self.calculate_sortino_ratio(self.benchmark_returns),
            self.calculate_sortino_ratio(self.strategy_returns),
            self.calculate_sortino_ratio(self.excess_returns)]

        df = pd.DataFrame(metrics)
        df.set_index(['benchmark', 'strategy', 'excess'], inplace=True)

        return df

    def plot_returns_distribution(self):
        """绘制收益分布直方图和统计指标"""
        plt.figure(figsize=(10, 6))

        # 基准收益分布
        plt.subplot(1, 2, 1)
        plt.hist(self.benchmark_returns, bins=30, alpha=0.5, label='Benchmark Returns')
        plt.axvline(self.benchmark_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.benchmark_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # 基准收益概率密度曲线
        plt.twinx()
        kde = gaussian_kde(self.benchmark_returns['returns'])
        x1 = np.linspace(np.min(self.benchmark_returns['returns']), np.max(self.benchmark_returns['returns']), 100)
        y1 = kde(x1)
        plt.plot(x1, y1, color='blue', label='KDE')
        plt.title('Benchmark Returns Distribution')
        plt.legend()

        # 策略收益分布
        plt.subplot(1, 2, 2)
        plt.hist(self.strategy_returns, bins=30, alpha=0.5, label='Strategy Returns')
        plt.axvline(self.strategy_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.strategy_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # 策略收益概率密度曲线
        plt.twinx()
        kde = gaussian_kde(self.strategy_returns['returns'])
        x2 = np.linspace(np.min(self.strategy_returns['returns']), np.max(self.strategy_returns['returns']), 100)
        y2 = kde(x2)
        plt.plot(x2, y2, color='blue', label='KDE')
        plt.title('Strategy Returns Distribution')
        plt.legend()

        # 超额收益分布
        plt.subplot(1, 2, 3)
        plt.hist(self.excess_returns, bins=30, alpha=0.5, label='Strategy Returns')
        plt.axvline(self.excess_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.excess_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # 超额收益概率密度曲线
        plt.twinx()
        kde = gaussian_kde(self.excess_returns['returns'])
        x3 = np.linspace(np.min(self.excess_returns['returns']), np.max(self.excess_returns['returns']), 100)
        y3 = kde(x3)
        plt.plot(x3, y3, color='blue', label='KDE')
        plt.title('Excess Returns Distribution')
        plt.legend()

        plt.tight_layout()
        plt.show()

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

        df = []
        for value in values_list:
            stats = {}
            stats['mean'] = np.round(np.mean(value), decimals=4)
            stats['std'] = np.round(np.std(value), decimals=4)
            stats['median'] = np.round(np.median(value), decimals=4)
            stats['kurtosis'] = np.round(kurtosis(value), decimals=4)
            stats['skewness'] = np.round(skew(value), decimals=4)

            df1 = pd.DataFrame(stats)
            df = df.append(df1)

        df.set_index(['benchmark', 'strategy', 'excess'], inplace=True)
        df.index.name = 'Stats'

        return df

    def run_backtest(self):
        """运行回测框架"""
        self.execute_trades()
        self.plot_performance()
        df = self.get_dataframe()
        print("Returns Dataframe:")
        print(df)
        self.plot_returns_distribution()


if __name__ == '__main__':
    # 回测起止时间
    start_date = pd.to_datetime('20140101')
    end_date = pd.to_datetime('20231208')

    # 交易标的和对应权重
    assets = ['000985.CSI']
    weights = [0.5]

    # 创建回测对象并执行回测
    backtester = CTABacktester(start_date, end_date, assets, weights)
    backtester.run_backtest()
