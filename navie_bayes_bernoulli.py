# 基于伯努利模型

from numpy.ma import ones, log, array


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


# 创建词表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 伯努利形式使用词集模式, 将输入的文档转化为词向量
def setOfWords2Vec(vocablist, inputSet):
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec


# 多项式模式使用词袋模式, 将输入的文档转化为词向量
def bagOfWords2VecMN(vocablist, inputSet):
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 总的训练文档数
    numWords = len(trainMatrix[0])  # 词表的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 伯努利模型先验概率 类型为1的文档 / 总训练文档
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):  # 伯努利模型的实现
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 统计该类别下每个单词出现过的文档数
        else:
            p0Num += trainMatrix[i]
    p0Denom += (numTrainDocs - sum(trainCategory))
    p1Denom += sum(trainCategory)
    p1Vect = log(p1Num / p1Denom)  # 伯努利模型条件概率 包含单词的文档数+1 / 类下文档总数+2
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    testingNB()
