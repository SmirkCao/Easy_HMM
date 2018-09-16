#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Easy_HMM
# Filename: test_stock
# Date: 9/16/18
# Author: 😏 <smirk dot cao at gmail dot com>
from examples.stock import *
import unittest
import logging


class TestWordseg(unittest.TestCase):
    def test_load_data(self):
        # 测试数据维度
        _, X = load_data('../data/yahoofinance-INTC-19950101-20040412.csv')
        X_shape = X.shape
        logger.info(X_shape)
        self.assertEqual(len(X_shape), 2)
        self.assertTupleEqual(X_shape, (2334, 2))

    def test_load_data_diff(self):
        #  测试两种数据读取方式结果一样
        _, X1 = load_data('../data/yahoofinance-INTC-19950101-20040412.csv')
        _, X2 = load_data('../data/yahoofinance-INTC-19950101-20040412.csv')
        self.assertSequenceEqual(X1.flatten().tolist(), X2.flatten().tolist())
        self.assertSequenceEqual([1, 2], [1, 2])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()
