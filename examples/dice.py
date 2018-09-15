#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Easy_HMM
# Filename: dice
# Date: 9/14/18
# Author: 😏 <smirk dot cao at gmail dot com>
from hmmlearn import hmm
import numpy as np
import logging
""" 
色子问题, 包含三个小问题, 已知观测色子大小序列X
A. 求每次丢的骰子的种类      对应求隐藏的状态序列
B. 丢出该串数字的概率        对应条件概率求和
C. 下次丢骰子最有可能的数字   这个例子稍微有点特殊, 因为转移矩阵是对称的, 各个状态跳转的概率是一样的, 看不出固定的结果.
"""
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # for this example
    # n_components: 3   对应三个色子
    # n_features: 1     理解一下, 这里feature和symbol是不一样的含义. symbol对应色子面的八种状态
    # 初始概率: 三个色子随机抽取
    startprob = np.ones(3)
    startprob /= startprob.sum()
    # 转移矩阵: 机会均等
    transmat = np.ones((3, 3))
    transmat /= transmat.sum(axis=1)
    # 观测矩阵: 和色子面数有关系
    emissionprob = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    emissionprob /= emissionprob.sum(axis=1, keepdims=True)

    hmmdice = hmm.MultinomialHMM(n_components=3, algorithm="map")
    hmmdice.startprob_ = startprob
    hmmdice.transmat_ = transmat
    hmmdice.emissionprob_ = emissionprob
    X = np.array([1, 6, 3, 5, 2, 7, 3, 5, 2, 4, 3, 6, 1, 5, 4]).reshape(-1, 1)
    # 效果一样
    # X = np.array([[1, 6, 3, 5, 2, 7, 3, 5, 2, 4, 3, 6, 1, 5, 4]]).T
    # 问题A
    prob, rst = hmmdice.decode(X)
    logger.info("\n%s" % hmmdice.startprob_)
    logger.info("\n%s" % hmmdice.transmat_)
    logger.info("\n%s" % hmmdice.emissionprob_)
    logger.info(hmmdice.predict(X))
    # 问题B
    logger.info(prob)
    logger.info(rst)
    prob = hmmdice.predict_proba(X)
    logger.info("\n%s" % prob)
    # 问题C
    hmmdice.startprob_ = prob[-1] >= prob[-1]
    rst, state = hmmdice.sample(15)
    logger.info("\n%s\n%s" % (rst.flatten()[1:], state[1:]))

    # Error using MultinomialHMM #154
    # X1 = np.array([0, 1, 1, 1, 1, 3, 1, 3, 1])
    # X2 = np.array([0, 2, 3, 1, 3, 2, 3])
    # X = np.concatenate([X1, X2])
    # lengths = [len(X1), len(X2)]
    # hmm_issue = hmm.MultinomialHMM(n_components=2)
    # hmm_issue.fit(X.reshape(-1, 1), lengths)
    # logger.info("\n%s" % hmm_issue.emissionprob_)
    # logger.info("\n%s" % hmm_issue.startprob_)
    # logger.info("\n%s" % hmm_issue.transmat_)
