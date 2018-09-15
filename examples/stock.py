#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Easy_HMM
# Filename: stock
# Date: 9/15/18
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.preprocessing import scale
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn import hmm

import pandas as pd
import numpy as np
import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    quotes = pd.read_csv('../data/yahoofinance-INTC-19950101-20040412.csv')

    dates = quotes.index.values
    close_v = quotes[["Close"]].values.flatten()
    volume = quotes[["Volume"]].values.flatten()
    # diff：out[n] = a[n+1] - a[n] 得到价格变化
    diff = np.diff(close_v)
    dates = dates[1:]
    close_v = close_v[1:]
    volume = volume[1:]

    # scale归一化处理：均值为0和方差为1
    # 将价格和交易数组成输入数据
    X = np.column_stack([scale(diff), scale(volume)])
    model = hmm.GaussianHMM(n_components=4)
    model.fit(X)
    # 预测隐状态
    _, hidden_states = model.decode(X)

    # 打印参数
    print("Transition matrix: ", model.transmat_)
    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", model.covars_[i])
        print()

    # 画图描述
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))

        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())

        ax.grid(True)

    plt.show()
