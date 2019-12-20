### 朴素贝叶斯
*最近阅读《机器学习实战》这本书中的朴素贝叶斯中的代码对一部分内容有点疑问，不知道是作者写错了还是我理解错了*

![navie_bayes.png](image/机器学习实战.png)

*作者上面计算先验概率是用的是伯努利模型，但是下面计算条件概率却使用多项式模型（也不是多项式，对于一个文档来说他只统计词有没有出现，而没统计出现多少次）
所以我把自己的认为正确的代码保存在这里以供以后再看*
* 计算公式：$p(c_i|w) = \frac{p(w|c_i)p(c_i)}{p(w)}$
* 其中分母 p(w) 在计算的时候都相同，所以只需要计算分子 $p(w|c_i)p(c_i)$
* 条件概率 $p(w|c_i) = p(w_0|c_i)p(w_1|c_i)...p(w_n|c_i)$

#### 多项式模式模型
* 多项式模型以词为粒度
* 先验概率 $p(c_i) = \frac{类c_i下所有文档单词总数}{所有训练样本文档的单词总数}$
* 条件概率 $p(w_j|c_i) = \frac{类c_i下单词w_j在所有文档中出现过的次数之和+1}{类c_i下单词总数+m}$

``
其中m是词表的长度，也就是总训练样本中所有不重复单词的数量
``

#### 伯努利模型
* 伯努利模型以文件为粒度
* 先验概率 $P(c_i) = \frac{类c_i下文档总数}{所有训练样本的文档总数}$
* 条件概率 $p(w_j|c_i) = \frac{类c_i下包含单词w_j的文档数+1}{类c_i下文档总数+2}$

#### 举个例子
编号|文档|类别(love?)
---|:--:|---:
1|I love china|yes
2|china is beautiful|yes
3|china china love|yes
4|stupid china|no
5|fuck china|no
```
词典 {I, love, china, is, beautiful, stupid, fuck} 
测试文档  china love stupid
```
* 多项式模式模型

$p(yes) = \frac{9}{9+4} = \frac{9}{13}$ $p(no) = \frac{4}{9+4} = \frac{4}{13}$

$p(china|yes) = \frac{4+1}{9+7} = \frac{5}{16}$

$p(love|yes) = \frac{2+1}{9+7} = \frac{3}{16}$

$p(stupid|yes) = \frac{0+1}{9+7} = \frac{1}{16}$

$p(china|no) = \frac{2+1}{4+7} = \frac{3}{11}$

$p(love|no) = \frac{0+1}{4+7} = \frac{1}{11}$

$p(stupid|no) = \frac{1+1}{4+7} = \frac{2}{11}$

$p(yes|china love stupid) = \frac{9}{13} * \frac{5}{16} * \frac{3}{16} * \frac{1}{16} = 0.002535$

$p(no|china love stupid) = \frac{4}{13} * \frac{3}{11} * \frac{1}{11} * \frac{2}{11} = 0.001387$

故该文档属于yes

* 伯努利模型

$p(yes) = \frac{3}{5}$ $p(no) = \frac{2}{5}$ 

$p(china|yes) = \frac{3+1}{3+2} = \frac{4}{5}$

$p(love|yes) = \frac{2+1}{3+2} = \frac{3}{5}$

$p(stupid|yes) = \frac{0+1}{3+2} = \frac{1}{5}$

$p(china|no) = \frac{2+1}{2+2} = \frac{3}{4}$

$p(love|no) = \frac{0+1}{2+2} = \frac{1}{4}$

$p(stupid|no) = \frac{1+1}{2+2} = \frac{2}{4}$

$p(yes|china love stupid) = \frac{3}{5} * \frac{4}{5} * \frac{3}{5} * \frac{1}{5} = 0.0576$

$p(no|china love stupid) = \frac{2}{5} * \frac{3}{4} * \frac{1}{4} * \frac{2}{4} = 0.0375$

故该文档属于yes