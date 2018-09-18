# Code Review



## Dice_01.py

这段代码用来分析色子(Dice), 已知三个色子, 给出一个观测序列, 分析ABC三个问题.

```python
# -*- coding:utf-8 -*-
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
import numpy as np
import hmm
```

- N: 可能的状态数, 三个色子, 对应三种隐状态, 取值空间[0, 1, 2]

- M: 可能的观测数, 色子面编号, 对应观测序列中可能出现的八种状态, 取值空间[0, 1, 2, 3, 4, 5, 6, 7]

```python
dice_num = 3
x_num = 8
```

- 离散马尔可夫模型
- 参数已知, 三个概率对应如下:
  - 初始状态概率分布: 随机
  - 状态转移概率: 随机, 这个例子比较简单, 隐藏状态之间的转换概率是均等的, 也就是已知.
  - 观测概率矩阵: 可以看出, 对应的色子为[**六面色, 四面色, 八面色**], 对应值为[0, 1, 2]

```python
dice_hmm = hmm.DiscreteHMM(3, 8)
dice_hmm.start_prob = np.ones(3) / 3.0
dice_hmm.transmat_prob = np.ones((3, 3)) / 3.0
dice_hmm.emission_prob = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
# 归一化
dice_hmm.emission_prob = dice_hmm.emission_prob / np.repeat(np.sum(dice_hmm.emission_prob, 1), 8).reshape((3, 8))
```

这里因为numpy的broadcast特性, 可以最后一句更改一下, 参考下面

```python
A/np.sum(A,axis=1, keepdims=True)
```

因为状态转移概率已经知道, 所以设置模型参数trainned为True

```python
dice_hmm.trained = True
```

已知观测序列的情况下, 通过模型可以求解三个问题:

```python
X = np.array([[1], [6], [3], [5], [2], [7], [3], [5], [2], [4], [3], [6], [1], [5], [4]])
```

上面这个代码直接定义(15,1)的nparray有点累, 可以通过reshape实现.

```python
X = np.array([1, 6, 3, 5, 2, 7, 3, 5, 2, 4, 3, 6, 1, 5, 4]).reshape(-1,1)
```

问题A
解码, 通过上面已知观测序列, 看对应的隐藏状态, 也就是说, 抛哪个色子得到的这个观测.

> 观测:obse.:  [1, 6, 3, 5, 2, 7, 3, 5, 2, 4, 3, 6, 1, 5, 4]
> 结果: state:  [1. 2. 1. 0. 1. 2. 1. 0. 1. 0. 1. 2. 1. 0. 0.]

这里有些位置是显然的, 6,7, 只出现在八面色中, 对应位置是2. 

```python
Z = dice_hmm.decode(X)  # 问题A
```

问题B

丢出该观测结果的概率

```python
logprob = dice_hmm.X_prob(X)  # 问题B
```

问题C

预测下一个观测值, 注意这段代码里面c并没有用到. 这段代码不是很清晰.

```python
# 问题C
x_next = np.zeros((x_num, dice_num))
for i in range(x_num):
    c = np.array([i])
    x_next[i] = dice_hmm.predict(X, i)
```

```python
print("state: ", Z)
print("logprob: ", logprob)
print("prob of x_next: ", x_next)
```

## Wordseg_02.py

这段代码用来实现中文分词, 语料用的是人民日报的语料, 这个语料是标注过的, 所以是个监督学习问题.

```python
# -*-coding:utf-8
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
import numpy as np
import hmm
```

以字为单位, 可以有四种状态, BEMS, 这是已知. 这里state_M从已知的状态获取可能更容易理解.

- $B\rightarrow Begin$
- $E\rightarrow End$
- $M\rightarrow Middle$
- $S\rightarrow Single$

```python
state_M = 4
word_N = 0

state_list = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
```

这个函数, 输入是已经分好的一个串, 输出是针对串做的标注.

```python
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
```

这个函数处理词典, 

```python
# 预处理词典：RenMinData.txt_utf8
def precess_data():
    ifp = open("RenMinData.txt_utf8")
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
        if not line: continue
        # line = line.decode("utf-8","ignore")

        word_list = []
        for i in range(len(line)):
            if line[i] == " ": continue
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
    ifp.close()

    lines = []
    for i in range(line_num):
        lines.append(np.array([[word_dic[x]] for x in line_seq[i]]))

    return lines, state_seq, word_dic
```



```python
# 将句子转换成字典序号序列
def word_trans(wordline, word_dic):
    word_inc = []
    line = wordline.strip()
    # line = line.decode("utf-8", "ignore")
    for n in range(len(line)):
        word_inc.append([word_dic[line[n]]])

    return np.array(word_inc)
```

lines, state_seq, word_dic:

- 这里有X, Z相当于是有了数据和对应的标签.

```python
X, Z, word_dic = precess_data()
```

解释下参数

- 第一个参数, 状态数量
- 第二个参数, 可能的观测值
- 第三个参数, 迭代轮数, 这里用了5, 跑起来时间已经很长了.

```python
wordseg_hmm = hmm.DiscreteHMM(4, len(word_dic), 5)
```

这里通过训练, 可以拿到转移矩阵. 以及先验.

这里用的是train_batch, 有相关的推导. 训练数据不是单一的序列, 而是多个短序列.
注意这里将分词标注的结果传递给模型训练, 本代码测试案例中, 训练1轮的结果和5轮是一样的.

```python
wordseg_hmm.train_batch(X, Z)

print("startprob_prior: ", wordseg_hmm.start_prob)
print("transmit: ", wordseg_hmm.transmat_prob)
```

训练之后, 可以实现给定句子, 完成分词. 

这里其实有点问题, 不能每次run这个code, 都要重新run一遍, 训练好的参数, 应该可以复用. 

```python

sentence_1 = "我要回家吃饭"
sentence_2 = "中国人民从此站起来了"
sentence_3 = "经党中央研究决定"
sentence_4 = "江主席发表重要讲话"

Z_1 = wordseg_hmm.decode(word_trans(sentence_1, word_dic))
Z_2 = wordseg_hmm.decode(word_trans(sentence_2, word_dic))
Z_3 = wordseg_hmm.decode(word_trans(sentence_3, word_dic))
Z_4 = wordseg_hmm.decode(word_trans(sentence_4, word_dic))

print(u"我要回家吃饭: ", Z_1)
print(u"中国人民从此站起来了: ", Z_2)
print(u"经党中央研究决定: ", Z_3)
print(u"江主席发表重要讲话: ", Z_4)
```

博客里面提到了另外一份代码, 里面实现了快速分词.

## Stock_03.py

代码给不同的股市状态做了标记(对应的隐状态), 可以区分出来不同时刻对应什么状态, 这个图里中间连接的比较直的部分, 不是实际的连线, 着重看的是**数据点**.

```python
# -*- coding:utf-8 -*-
# By tostq <tostq216@163.com>
# Reference to hmmlearn.examples.plot_hmm_stock_analysis.py
# 博客: blog.csdn.net/tostq
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmm import GaussianHMM
from sklearn.preprocessing import scale
```

这段和原始版本的code 有所更改, 因为matplotlib2.0取消了finance模块. 看到这其实大概明白了为什么会有很多的算法实现会标记不依赖任何库, 其实, 意义也不大, 即便只依赖python, 如果是2.0版本的code, 那到了py3也一样.

- diff, 感觉和pandas里面的shift差不多, 没仔细看. 还有就是在金融数据分析里面常用对数收益, 这个代码没有体现.
- 源代码中volume没有选子集, 酱紫会长度不匹配, 没仔细看直接添加了volume[1:]

```python
###############################################################################
# 导入Yahoo金融数据
quotes = pd.read_csv('data/yahoofinance-INTC-19950101-20040412.csv')

dates = quotes.index.values
close_v = quotes[["Close"]].values.flatten()
volume = quotes[["Volume"]].values.flatten()
# diff：out[n] = a[n+1] - a[n] 得到价格变化
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]
volume = volume[1:]
```

其实用了pandas, 这里有些操作是多余的.比如这个等价于np.vstack((a,b)).T的操作

```python
# scale归一化处理：均值为0和方差为1
# 将价格和交易数组成输入数据
X = np.column_stack([scale(diff), scale(volume)])
```

这里有几个问题:

1. 为什么隐藏的状态是4?
   其实我们不知道状态是什么, 大概估计下, 算了4. 还有给6种状态的.
1. 为什么用高斯HMM模型?
   这是个假设, 假设随机变量服从高斯分布.

```python
# 训练高斯HMM模型，这里假设隐藏状态4个
model = GaussianHMM(4,2,20)
model.train(X)
```

```python
# 预测隐状态
hidden_states = model.decode(X)
```

看了三个例子, 显然知道训练的过程我们拿到了转移矩阵

高斯分布, 比较好的一点是我们只需要存储模型的均值和方差, 就可以通过均值和方差来复原联合分布.

```python
# 打印参数
print("Transition matrix: ", model.transmat_prob)
print("Means and vars of each hidden state")
for i in range(model.n_state):
    print("{0}th hidden state".format(i))
    print("mean = ", model.emit_means[i])
    print("var = ", model.emit_covars[i])
    print()
```

这个画图就是让你看到, 哪些时间段对应了哪些状态.

```python
# 画图描述
fig, axs = plt.subplots(model.n_state, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_state))
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
```

## hmm.py

这个代码好长, 单独开一个好像更合适. 粗略了看下, 应该能更短啊.  

文件总长度是365行, 按照一天一行的节奏,一年也就搞定了.:smirk:

### _BaseHMM

```python
# -*- coding:utf-8 -*-
# 隐马尔科夫链模型
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
import numpy as np
from math import pi, sqrt, exp, pow
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster
```

```python
class _BaseHMM():
    """
    基本HMM虚类，需要重写关于发射概率的相关虚函数
    n_state : 隐藏状态的数目
    n_iter : 迭代次数
    x_size : 观测值维度
    start_prob : 初始概率
    transmat_prob : 状态转换概率
    """
    __metaclass__ = ABCMeta  # 虚类声明
```

- 默认的初始状态概率是均等的
- 默认的个状态之间的转移矩阵也是均等的
- 默认迭代次数20次

```python
def __init__(self, n_state=1, x_size=1, iter=20):
    self.n_state = n_state
    self.x_size = x_size
    self.start_prob = np.ones(n_state) * (1.0 / n_state)  # 初始状态概率
    self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)  # 状态转换概率矩阵
    self.trained = False  # 是否需要重新训练
    self.n_iter = iter  # EM训练的迭代次数
```

对于不同的马尔可夫模型, 有些函数需要重载, 这里面提到的发射概率参考PRML里面的描述, 某一个状态下产生不同观测的条件概率.

- 发射概率
- 发射概率更新

```python
# 初始化发射参数
@abstractmethod
def _init(self, X):
    pass

# 虚函数：返回发射概率
@abstractmethod
def emit_prob(self, x):  # 求x在状态k下的发射概率 P(X|Z)
    return np.array([0])

# 虚函数
@abstractmethod
def generate_x(self, z):  # 根据隐状态生成观测值x p(x|z)
    return np.array([0])

# 虚函数：发射概率的更新
@abstractmethod
def emit_prob_updated(self, X, post_state):
    pass
```

- 序列生成, 其实就是采样.
- 参数是序列长度, 这里需要生成多少的序列, 就生成多少.
- 按照转移矩阵跳转这个具体实现通过np.random.choice实现, 这个函数有个参数replace=True, 默认有放回
- Z_pre用来实现状态跳转矩阵中的状态选择
- 返回采样结果(观测序列)以及对应的隐藏状态序列, 观测和隐藏状态是一一对应的, 这个是观测独立性假设.

```python
# 通过HMM生成序列
def generate_seq(self, seq_length):
    X = np.zeros((seq_length, self.x_size))
    Z = np.zeros(seq_length)
    Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)  # 采样初始状态
    X[0] = self.generate_x(Z_pre)  # 采样得到序列第一个值
    Z[0] = Z_pre

    for i in range(seq_length):
        if i == 0: continue
        # P(Zn+1)=P(Zn+1|Zn)P(Zn)
        Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre, :][0])
        Z_pre = Z_next
        # P(Xn+1|Zn+1)
        X[i] = self.generate_x(Z_pre)
        Z[i] = Z_pre
    return X, Z
```

- 如果隐藏状态是已知的, 那这个事相对好干很多, 查表连乘应该就可以了.
- Z_seq有任何一个值, 都认为隐藏状态已知
- 只用到了向前传递
- 返回的概率已经是对数和的形式了

```python
# 估计序列X出现的概率
def X_prob(self, X, Z_seq=np.array([])):
    # 状态序列预处理
    # 判断是否已知隐藏状态
    X_length = len(X)
    if Z_seq.any():
        Z = np.zeros((X_length, self.n_state))
        for i in range(X_length):
            Z[i][int(Z_seq[i])] = 1
    else:
        Z = np.ones((X_length, self.n_state))
    # 向前向后传递因子
    _, c = self.forward(X, Z)  # P(x,z)
    # 序列的出现概率估计
    prob_X = np.sum(np.log(c))  # P(X)
    return prob_X
```

- forward, 前面的用到c, 这个用到alpha

```python
# 已知当前序列预测未来（下一个）观测值的概率
def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
    if self.trained == False or istrain == False:  # 需要根据该序列重新训练
        self.train(X)
    X_length = len(X)
    if Z_seq.any():
        Z = np.zeros((X_length, self.n_state))
        for i in range(X_length):
            Z[i][int(Z_seq[i])] = 1
    else:
        Z = np.ones((X_length, self.n_state))
        # 向前向后传递因子
        alpha, _ = self.forward(X, Z)  # P(x,z)
        prob_x_next = self.emit_prob(np.array([x_next])) * np.dot(alpha[X_length - 1], 
                                     self.transmat_prob)
    return prob_x_next
```

- 其实如果没有train就直接train这个操作也有点用力过猛感觉.
- 给定观测序列, 输出最有可能的对应的状态序列, 这个是HMM的三个基本问题的预测问题.
- 这块代码应该不是维特比

```python
def decode(self, X, istrain=True):
    """
        利用维特比算法，已知序列求其隐藏状态值
        :param X: 观测值序列
        :param istrain: 是否根据该序列进行训练
        :return: 隐藏状态序列
        """
    if self.trained == False or istrain == False:  # 需要根据该序列重新训练
        self.train(X)

    X_length = len(X)  # 序列长度
    state = np.zeros(X_length)  # 隐藏状态

    pre_state = np.zeros((X_length, self.n_state))  # 保存转换到当前隐藏状态的最可能的前一状态
    max_pro_state = np.zeros((X_length, self.n_state))  # 保存传递到序列某位置当前状态的最大概率

    _, c = self.forward(X, np.ones((X_length, self.n_state)))
    # 初始概率
    max_pro_state[0] = self.emit_prob(X[0])*self.start_prob*(1 / c[0])  

    # 前向过程
    for i in range(X_length):
        if i == 0: continue
        for k in range(self.n_state):
            prob_state = self.emit_prob(X[i])[k]*\
                         self.transmat_prob[:, k]*\
                         max_pro_state[i - 1]
            max_pro_state[i][k] = np.max(prob_state) * (1 / c[i])
            pre_state[i][k] = np.argmax(prob_state)
    # 后向过程
    state[X_length - 1] = np.argmax(max_pro_state[X_length - 1, :])
    for i in reversed(range(X_length)):
        if i == X_length - 1: continue
        state[i] = pre_state[i + 1][int(state[i + 1])]
            
    return state
```

- 1 to k参考PRML从二项分布推广到多项分布的部分有描述. 和One Hot是一个形式. 如果这里用pandas的话会更简单.

```python
# 针对于多个序列的训练问题
def train_batch(self, X, Z_seq=list()):
    # 针对于多个序列的训练问题，其实最简单的方法是将多个序列合并成一个序列，而唯一需要调整的是初始状态概率
    # 输入X类型：list(array)，数组链表的形式
    # 输入Z类型: list(array)，数组链表的形式，默认为空列表（即未知隐状态情况）
    self.trained = True
    X_num = len(X)  # 序列个数
    self._init(self.expand_list(X))  # 发射概率的初始化

    # 状态序列预处理，将单个状态转换为1-to-k的形式
    # 判断是否已知隐藏状态
    if Z_seq == list():
        Z = []  # 初始化状态序列list
        for n in range(X_num):
            Z.append(list(np.ones((len(X[n]), self.n_state))))
    else:
        Z = []
        for n in range(X_num):
            Z.append(np.zeros((len(X[n]), self.n_state)))
            for i in range(len(Z[n])):
                Z[n][i][int(Z_seq[n][i])] = 1
                
    for e in range(self.n_iter):  # EM步骤迭代
        # 更新初始概率过程
        #  E步骤
        print("iter: ", e)
        # 批量累积：状态的后验概率，类型list(array)
        b_post_state = []  
        # 批量累积：相邻状态的联合后验概率，数组
        b_post_adj_state = np.zeros((self.n_state, self.n_state))  
        b_start_prob = np.zeros(self.n_state)  # 批量累积初始概率
        for n in range(X_num):  # 对于每个序列的处理
            X_length = len(X[n])
            alpha, c = self.forward(X[n], Z[n])  # P(x,z)
            beta = self.backward(X[n], Z[n], c)  # P(x|z)

            post_state = alpha * beta / np.sum(alpha * beta)  # 归一化！
            b_post_state.append(post_state)
            # 相邻状态的联合后验概率
            post_adj_state = np.zeros((self.n_state, self.n_state))  
            for i in range(X_length):
                if i == 0: continue
                if c[i] == 0: continue
                post_adj_state += (1 / c[i]) * \
                                  np.outer(alpha[i - 1],
                                           beta[i] * self.emit_prob(X[n][i])) *\
                                  self.transmat_prob

            if np.sum(post_adj_state) != 0:
                post_adj_state = post_adj_state / np.sum(post_adj_state)  # 归一化！
            b_post_adj_state += post_adj_state  # 批量累积：状态的后验概率
            b_start_prob += b_post_state[n][0]  # 批量累积初始概率

        # M步骤，估计参数，最好不要让初始概率都为0出现，这会导致alpha也为0
        b_start_prob += 0.001 * np.ones(self.n_state)
        self.start_prob = b_start_prob / np.sum(b_start_prob)
        b_post_adj_state += 0.001
        for k in range(self.n_state):
            if np.sum(b_post_adj_state[k]) == 0: continue
            self.transmat_prob[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])
        self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))
```

- 展开, 这个操作有点搞复杂了

```python
def expand_list(self, X):
        # 将list(array)类型的数据展开成array类型
        C = []
        for i in range(len(X)):
            C += list(X[i])
    return np.array(C)
```

- 单个序列训练, EM算法

```python
# 针对于单个长序列的训练
def train(self, X, Z_seq=np.array([])):
    # 输入X类型：array，数组的形式
    # 输入Z类型: array，一维数组的形式，默认为空列表（即未知隐状态情况）
    self.trained = True
    X_length = len(X)
    self._init(X)

    # 状态序列预处理
    # 判断是否已知隐藏状态
    if Z_seq.any():
        Z = np.zeros((X_length, self.n_state))
        for i in range(X_length):
            Z[i][int(Z_seq[i])] = 1
    else:
        Z = np.ones((X_length, self.n_state))
        
    for e in range(self.n_iter):  # EM步骤迭代
        # 中间参数
        print(e, " iter")
        # E步骤
        # 向前向后传递因子
        alpha, c = self.forward(X, Z)  # P(x,z)
        beta = self.backward(X, Z, c)  # P(x|z)
        
        post_state = alpha * beta
        post_adj_state = np.zeros((self.n_state, self.n_state))  # 相邻状态的联合后验概率
        for i in range(X_length):
            if i == 0: continue
            if c[i] == 0: continue
            post_adj_state += (1 / c[i]) *\
                               np.outer(alpha[i - 1],                                                                           beta[i] *\
                               self.emit_prob(X[i])) *\
                               self.transmat_prob

        # M步骤，估计参数
        self.start_prob = post_state[0] / np.sum(post_state[0])
        for k in range(self.n_state):
            self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])
            
        self.emit_prob_updated(X, post_state)
```

- forward, 求$\alpha$

```python
# 求向前传递因子
def forward(self, X, Z):
    X_length = len(X)
    alpha = np.zeros((X_length, self.n_state))  # P(x,z)
    # 初始值
    alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0]  
    # 归一化因子
    c = np.zeros(X_length)
    c[0] = np.sum(alpha[0])
    alpha[0] = alpha[0] / c[0]
    # 递归传递
    for i in range(X_length):
        if i == 0: continue
        alpha[i] = self.emit_prob(X[i]) *\
                   np.dot(alpha[i - 1], self.transmat_prob) *\
                   Z[i]
        c[i] = np.sum(alpha[i])
        if c[i] == 0: continue
        alpha[i] = alpha[i] / c[i]
    return alpha, c
```

- backward, 求$\beta$

```python
# 求向后传递因子
def backward(self, X, Z, c):
    X_length = len(X)
    beta = np.zeros((X_length, self.n_state))  # P(x|z)
    beta[X_length - 1] = np.ones((self.n_state))
    # 递归传递
    for i in reversed(range(X_length)):
        if i == X_length - 1: continue
        beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]),
                         self.transmat_prob.T) *\
                  Z[i]
        if c[i + 1] == 0: continue
        beta[i] = beta[i] / c[i + 1]
    return beta
```

### Gaussian2D

- 这里很多操作都可以用numpy实现

```python
# 二元高斯分布函数
def gauss2D(x, mean, cov):
    # x, mean, cov均为numpy.array类型
    z = -np.dot(np.dot((x - mean).T, inv(cov)), (x - mean)) / 2.0
    temp = pow(sqrt(2.0 * pi), len(x)) * sqrt(det(cov))
    return (1.0 / temp) * exp(z)
```

### GaussianHMM

```python
class GaussianHMM(_BaseHMM):
    """
    发射概率为高斯分布的HMM
    参数：
    emit_means: 高斯发射概率的均值
    emit_covars: 高斯发射概率的方差
    """

    def __init__(self, n_state=1, x_size=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=x_size, iter=iter)
        self.emit_means = np.zeros((n_state, x_size))  # 高斯分布的发射概率均值
        self.emit_covars = np.zeros((n_state, x_size, x_size))  # 高斯分布的发射概率协方差
        for i in range(n_state): 
            self.emit_covars[i] = np.eye(x_size)  # 初始化为均值为0，方差为1的高斯分布函数

    def _init(self, X):
        # 通过K均值聚类，确定状态初始值
        mean_kmeans = cluster.KMeans(n_clusters=self.n_state)
        mean_kmeans.fit(X)
        self.emit_means = mean_kmeans.cluster_centers_
        for i in range(self.n_state):
            self.emit_covars[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))
```

```python
def emit_prob(self, x):  # 求x在状态k下的发射概率
    prob = np.zeros((self.n_state))
    for i in range(self.n_state):
        prob[i] = gauss2D(x, self.emit_means[i], self.emit_covars[i])
    return prob
```

```python
def generate_x(self, z):  # 根据状态生成x p(x|z)
    return np.random.multivariate_normal(self.emit_means[z][0], 
                                         self.emit_covars[z][0], 
                                         1)
```



```python
def emit_prob_updated(self, X, post_state):  # 更新发射概率
    for k in range(self.n_state):
        for j in range(self.x_size):
            self.emit_means[k][j] = np.sum(post_state[:, k] * \
                                           X[:, j]) / \
            							   np.sum(post_state[:, k])
            X_cov = np.dot((X - self.emit_means[k]).T, 
                           (post_state[:, k] * (X - self.emit_means[k]).T).T)
            self.emit_covars[k] = X_cov / np.sum(post_state[:, k])
            if det(self.emit_covars[k]) == 0:  # 对奇异矩阵的处理
                self.emit_covars[k] = self.emit_covars[k] + 0.01 * np.eye(len(X[0]))
```

### DiscreteHMM

不同的HMM差异在发射概率, 就是每个状态产生观测变量的概率分布.

```python
class DiscreteHMM(_BaseHMM):
    """
    发射概率为离散分布的HMM
    参数：
    emit_prob : 离散概率分布
    x_num：表示观测值的种类
    此时观测值大小x_size默认为1
    """

    def __init__(self, n_state=1, x_num=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=1, iter=iter)
        # 初始化发射概率均值
        self.emission_prob = np.ones((n_state, x_num)) * (1.0 / x_num)  
        self.x_num = x_num

    def _init(self, X):
        self.emission_prob = np.random.random(size=(self.n_state, self.x_num))
        for k in range(self.n_state):
            self.emission_prob[k] = self.emission_prob[k]/p.sum(self.emission_prob[k])
```

```python
def emit_prob(self, x):  # 求x在状态k下的发射概率
    prob = np.zeros(self.n_state)
    for i in range(self.n_state): 
        prob[i] = self.emission_prob[i][int(x[0])]
    return prob
```

```python
def generate_x(self, z):  # 根据状态生成x p(x|z)
    return np.random.choice(self.x_num, 1, p=self.emission_prob[z][0])
```

```python
def emit_prob_updated(self, X, post_state):  # 更新发射概率
    self.emission_prob = np.zeros((self.n_state, self.x_num))
    X_length = len(X)
    for n in range(X_length):
        self.emission_prob[:, int(X[n])] += post_state[n]
    self.emission_prob += 0.1 / self.x_num
    for k in range(self.n_state):
        if np.sum(post_state[:, k]) == 0: continue
        self.emission_prob[k] = self.emission_prob[k]/np.sum(post_state[:, k])
```

