#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Easy_HMM
# Filename: e10_2
# Date: 9/17/18
# Author: 😏 <smirk dot cao at gmail dot com>
from hmmlearn import hmm
import numpy as np
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    Q = {1: 0, 2: 1, 3: 2}
    V = {"red": 0, "white": 1}
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    T = 3
    O = ["red", "white", "red"]

    model_hmm = hmm.MultinomialHMM(n_components=3)
    model_hmm.startprob_ = pi
    model_hmm.transmat_ = A
    model_hmm.emissionprob_ = B

    prob, states = model_hmm.decode(np.array([V[x] for x in O]).reshape(-1, 1))
    score = model_hmm.score(np.array([V[x] for x in O]).reshape(-1, 1))

    logger.info("prob: %s hidden states: %s score: %s" % (prob, states, score))
    # np.log(0.13022) = --2.038529951173421
