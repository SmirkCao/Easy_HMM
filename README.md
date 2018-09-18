# Easy_HMM
A easy HMM program written with Python, including the full codes of training, prediction and decoding.

# Introduction
- Simple algorithms and models to learn HMMs in pure Python
- Including two HMM models: HMM with Gaussian emissions, and HMM with multinomial (discrete) emissions
- Using unnitest to verify our performance with [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/ "hmmlearn") . 
- Three examples: Dice problem, Chinese words segmentation and stock analysis.

# Code list
- hmm.py: hmm models file
- DiscreteHMM_test.py, GaussianHMM_test.py: test files
- Dice_01.py, Wordseg_02.py, Stock_03.py: example files
- RenMinData.txt_utf8: Chinese words segmentation datas

# 中文说明
参见个人博客：[http://blog.csdn.net/tostq/article/details/70846702](http://blog.csdn.net/tostq/article/details/70846702 "hmm")

里面具体剖析了HMM模型，这个代码也是上述系列博客的配套代码！

----
# 修改说明
1. 添加对Python 3的支持
1. 整理代码格式
1. 做了简单的[Code Review](./cr/code_review.md), 方便理解代码
1. 重新整理了目录结构, 并添加了新的代码实现
    1. cr       Code Review内容, 主要就是梳理easyhmm的代码
    1. data     保存数据, 人民日报数据和财经数据, 都是例子里面用的
    1. easyhmm  原repo里面的程序, 程序内容有更新, 可以看更新记录
    1. examples 重新整理的部分例子实现, 与原项目例子对应提供了三个程序通过hmmlearn实现
    1. models   原项目中添加了模型存储, 新示例程序也实现了模型存储, 存储位置在该目录
    1. test     新增测试案例, 实现demo程序的时候, 设计了一些用于功能实现的测试
1. 增加一个实例来自《统计机器学习》e10.2 e10.3, 这两个例子相对简单

总结几点:    
1. 这个项目比较适合入门HMM
1. 配合实例的算法描述容易弥补知识结构中比较小的gap, 实例的表达能力更强, 能够提供比文字以及图表的描述更多的信息. 
1. 文字->图表->实例->代码实例, 对于问题的理解与深入处在不同的维度. 
1. 感叹数学真的美妙, 做了层层压缩把极其丰富的信息压缩到文字表述中. 