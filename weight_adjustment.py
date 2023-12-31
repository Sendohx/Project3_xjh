# -*- coding = utf-8 -*-
# @Time: 2023/12/13 13:52
# @Author: Jiahao Xu
# @File：weight_adjustment.py
# @Software: PyCharm

import numpy as np


def weight_adjustment(asset_data):
    """
    权重调整函数
    : param asset_data: 标的dataframe
    """
    for i in range(len(asset_data) - 1):
        signal = asset_data.loc[i, 'ILLIQ_P']
        profit = asset_data.loc[i, 'return']
        position = asset_data.loc[i, 'weight']

        if signal > 65.0 and profit > 0:
            asset_data.loc[i + 1, 'weight'] = np.round(min(position + 0.1, 1.0), 1)
        elif signal > 65.0 and profit < 0:
            asset_data.loc[i + 1, 'weight'] = np.round(max(position - 0.1, 0.0), 1)
        else:
            asset_data.loc[i + 1, 'weight'] = np.round(position, 1)

    asset_data['weighted_return'] = asset_data['return'] * asset_data['weight']

    return asset_data

def weight_describe(asset_data):
    """
    计算仓位均值及调仓频率
    :param asset_data
    """
    mean_weight = asset_data['weight'].mean()

    position_changes = asset_data['weight'].diff() != 0
    adjustment_frequency = position_changes.sum() / len(asset_data)

    return f"仓位均值: {mean_weight}, 调仓频率: {adjustment_frequency}"
