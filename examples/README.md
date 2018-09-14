# Examples
[TOC]
## 前言

### hmmlearn中有些符号定义

- n_features	(int) Number of possible symbols emitted by the model (in the samples).
- monitor_	(ConvergenceMonitor) Monitor object used to check the convergence of EM.
- transmat_	(array, shape (n_components, n_components)) Matrix of transition probabilities between states.
- startprob_	(array, shape (n_components, )) Initial state occupation distribution.
- emissionprob_	(array, shape (n_components, n_features)) Probability of emitting a given symbol when in each state.

在hmmlearn里面用了feature这个概念, 对应了观测的集合.

### hmmlearn中函数说明

- decode: Find most likely state sequence corresponding to X
- fit 
- predict Find most likely state sequence corresponding to X.
- predict_proba
- sample:   Generate random samples from the model.
- score
- score_samples

### hmmlearn依赖

- 之前这个包是在sklearn中的, 有很多对sklearn的依赖, 现在依然有

## 示例

### 色子

todo: 补充下问题描述

这个例子对应了原来的Dice_01, 有几点说明:

1. 在之前的Code Review[^1]的时候提到过问题C部分不明确. 
1. 问题C的结果, 当前的状态, 对下一个结果有一定的影响, 当前隐状态是确定的了.所以重新初始化了值. 预测之后去掉了第一个值, 这里注意下.
1. 针对这个问题decode的结果和predict的结果一样.
1. 使用hmmlearn注意X的构建.
1. 代码的最后付了一段不同长度的sample输入的例子, 对应回答了在hmmlearn上的一个issue #154.



----

[^1]: [CR_Dice](../cr/code_review.md)