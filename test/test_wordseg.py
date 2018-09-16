#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Easy_HMM
# Filename: test_wordseg
# Date: 9/16/18
# Author: 😏 <smirk dot cao at gmail dot com>
from examples.wordseg import *
import unittest


class TestWordseg(unittest.TestCase):
    def test_get_bmes(self):
        state_list = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

        def decode(str_in):
            return [state_list[x] for x in str_in]
        self.assertEqual(get_bmes("中国"), decode("BE"))
        self.assertEqual(get_bmes("我"), decode("S"))
        self.assertEqual(get_bmes("秦皇岛"), decode("BME"))
        self.assertEqual(get_bmes(""), [])


if __name__ == '__main__':
    unittest.main()
