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
import warnings


def load_data(path):
    quotes = pd.read_csv(path, parse_dates=['Date'])
    quotes["Close_diff"] = quotes["Close"].diff()
    X = np.column_stack([scale(quotes["Close_diff"][1:]), scale(quotes["Volume"][1:])])
    return quotes, X


def load_data_old(path):
    quotes = pd.read_csv(path)
    dates = quotes.index.values
    close_v = quotes[["Close"]].values.flatten()
    volume = quotes[["Volume"]].values.flatten()
    diff = np.diff(close_v)

    dates = dates[1:]
    close_v = close_v[1:]
    volume = volume[1:]

    # scale归一化处理：均值为0和方差为1
    # 将价格和交易数组成输入数据
    X = np.column_stack([scale(diff), scale(volume)])
    return quotes, X


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    quotes, X = load_data('../data/yahoofinance-INTC-19950101-20040412.csv')
    model = hmm.GaussianHMM(n_components=4)
    model.fit(X)
    # 预测隐状态
    _, hidden_states = model.decode(X)

    # 打印参数
    logger.info("Transition matrix: %s" % model.transmat_)
    logger.info("Means and vars of each hidden state")
    for i in range(model.n_components):
        logger.info("{0}th hidden state".format(i))
        logger.info("mean = %s" % model.means_[i])
        logger.info("var = %s" % model.covars_[i])

    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(quotes[1:]["Date"][mask], quotes[1:]["Close"][mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))

        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())

        ax.grid(True)

    plt.show()
