#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
# Project:  Easy_HMM
# Filename: wordseg
# Date: 9/15/18
# Author: 😏 <smirk dot cao at gmail dot com>
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np

import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    state_M = 4
    word_N = 0

    state_list = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


    # 获得某词的分词结果
    # 如：（我：S）、（你好：BE）、（恭喜发财：BMME）
    def getList(input_str):
        outpout_str = []
        if len(input_str) == 1:
            outpout_str.append(3)
        elif len(input_str) == 2:
            outpout_str = [0, 2]
        else:
            M_num = len(input_str) - 2
            M_list = [1] * M_num
            outpout_str.append(0)
            outpout_str.extend(M_list)
            outpout_str.append(2)
        return outpout_str


    # 预处理词典：RenMinData.txt_utf8
    def precess_data():
        ifp = open("../data/RenMinData.txt_utf8")
        line_num = 0
        word_dic = {}
        word_ind = 0
        line_seq = []
        state_seq = []
        # 保存句子的字序列及每个字的状态序列，并完成字典统计
        for line in ifp:
            line_num += 1
            if line_num % 10000 == 0:
                print(line_num)

            line = line.strip()
            if not line:
                continue
            # line = line.decode("utf-8","ignore")

            word_list = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                word_list.append(line[i])
                # 建立单词表
                if not word_dic.__contains__(line[i]):
                    word_dic[line[i]] = word_ind
                    word_ind += 1
            line_seq.append(word_list)

            lineArr = line.split(" ")
            line_state = []
            for item in lineArr:
                line_state += getList(item)
            state_seq.append(np.array(line_state))
            # if line_num > 100:
            #     break
        ifp.close()

        lines = []
        [lines.extend(x) for x in line_seq]
        lines = np.array([word_dic[x] for x in lines])
        # for i in range(line_num):
        #     lines = np.vstack([lines, np.array([[word_dic[x]] for x in line_seq[i]])])

        return lines, state_seq, word_dic


    # 将句子转换成字典序号序列
    def word_trans(wordline, word_dic):
        word_inc = []
        line = wordline.strip()
        # line = line.decode("utf-8", "ignore")
        for n in range(len(line)):
            word_inc.append([word_dic[line[n]]])

        return np.array(word_inc)


    X, Z, word_dic = precess_data()
    joblib.dump(word_dic, "wordseg_hmm_dict.pkl")
    word_dic = joblib.load("wordseg_hmm_dict.pkl")
    logger.info("training start")
    # todo:分词效果不好, 重新看下
    wordseg_hmm = hmm.MultinomialHMM(4, len(word_dic), verbose=True, n_iter=6)
    lengths = [len(x) for x in Z]
    wordseg_hmm.fit(X.reshape(-1, 1), lengths=lengths)
    logger.info("training done")
    joblib.dump(wordseg_hmm, "wordseg_hmm.pkl")
    wordseg_hmm = joblib.load("wordseg_hmm.pkl")
    sentence_1 = "我要回家吃饭"
    sentence_2 = "中国人民从此站起来了"
    sentence_3 = "经党中央研究决定"
    sentence_4 = "江主席发表重要讲话"
    Z_1_prob, Z_1 = wordseg_hmm.decode(word_trans(sentence_1, word_dic))
    Z_2_prob, Z_2 = wordseg_hmm.decode(word_trans(sentence_2, word_dic))
    Z_3_prob, Z_3 = wordseg_hmm.decode(word_trans(sentence_3, word_dic))
    Z_4_prob, Z_4 = wordseg_hmm.decode(word_trans(sentence_4, word_dic))

    logger.info("我要回家吃饭: %s" % Z_1)
    logger.info("中国人民从此站起来了: %s" % Z_2)
    logger.info("经党中央研究决定: %s" % Z_3)
    logger.info("江主席发表重要讲话: %s" % Z_4)
