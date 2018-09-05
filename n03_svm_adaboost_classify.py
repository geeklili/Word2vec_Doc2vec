# coding: utf-8
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import warnings
import sys
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec


CUR_FILE = '.'

# 用gensim去做word2vec的处理，用sklearn当中的SVM进行建模

#  载入数据，做预处理(分词)，切分训练集与测试集, x: 分词后的文本列表，y:正向或者负向，1/0
def load_file_and_preprocessing():
    pos = pd.DataFrame([i.strip() for i in open(CUR_FILE + '/data/dev_other_data/dev02.txt', encoding='utf-8')])
    neg = pd.DataFrame([i.strip() for i in open(CUR_FILE + '/data/dev_other_data/other2.txt', encoding='utf-8')])

    # pos = pd.read_excel('./data/dev_other_data/neg.xls', header=None, index=None)
    # neg = pd.read_excel('./data/dev_other_data/pos.xls', header=None, index=None)

    # 新增一列 word ,存放分好词的评论，pos[0]代表表格第一列, pos['words']就成了第二列
    # apply可以把dataframe的一列或几列遍历计算
    pos['words'] = pos[0].apply(lambda x: list(jieba.cut(x)))
    neg['words'] = neg[0].apply(lambda x: list(jieba.cut(x)))

    # np.ones(len(pos)) 新建一个长度为len(pos)的数组并初始化元素全为1来标注好评
    # np.concatenate（）连接数组
    # axis=0 向下竖向合并 axis=1向右横向合并
    x = np.concatenate((pos['words'], neg['words']))
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
    print(x)

    # train_test_split：从样本中随机的按比例选取train data和testdata
    # 一般形式：train_test_split(train_data,train_target,test_size=0.4, random_state=0)
    # train_data：所要划分的样本特征集
    # train_target：所要划分的样本结果（标注）
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子。

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    np.save(CUR_FILE + '/data/svm_data/y_train.npy', y_train)
    np.save(CUR_FILE + '/data/svm_data/y_test.npy', y_test)
    return x_train, x_test


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text_li, size, w2v_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text_li:
        try:
            vec += w2v_model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vectors(x_train, x_test):
    n_dim = 100
    # 初始化模型和词表
    w2v_model = Word2Vec(x_train, size=n_dim, min_count=10)
    # hao = w2v_model['好评']
    # print(hao)
    # w2v_model = Word2Vec(size=100, window=5, min_count=10, workers=12)
    # w2v_model.build_vocab(x_train)
    #
    # w2v_model.train(x_train,
    #                total_examples=w2v_model.corpus_count,
    #                epochs=w2v_model.iter)

    # 获取到每行句子的合成向量，对每个句子的所有词向量取均值，来生成一个句子的vector
    train_vectors = np.concatenate([build_sentence_vector(i, n_dim, w2v_model) for i in x_train])
    # train_vectors = scale(train_vectors)
    # print(train_vectors)

    np.save(CUR_FILE + '/data/svm_data/train_vectors.npy', train_vectors)
    print(train_vectors.shape)

    # 在测试集上训练，增量训练
    # w2v_model.train(x_test)
    w2v_model.train(x_test, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)

    w2v_model.save(CUR_FILE + '/data/svm_data/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vectors = np.concatenate([build_sentence_vector(z, n_dim, w2v_model) for z in x_test])
    # test_vectors = scale(test_vectors)
    np.save(CUR_FILE + '/data/svm_data/test_vectors.npy', test_vectors)
    print(test_vectors.shape)


def get_data():
    train_vectors = np.load(CUR_FILE + '/data/svm_data/train_vectors.npy')
    y_train = np.load(CUR_FILE + '/data/svm_data/y_train.npy')
    test_vectors = np.load(CUR_FILE + '/data/svm_data/test_vectors.npy')
    y_test = np.load(CUR_FILE + '/data/svm_data/y_test.npy')
    return train_vectors, y_train, test_vectors, y_test


# 训练svm模型
def svm_train(train_vectors, train_y, test_vectors, test_y):
    # clf = SVC(kernel='rbf', verbose=True)
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(train_vectors, train_y)
    joblib.dump(clf, CUR_FILE + '/data/svm_data/model.pkl')
    print(clf.score(test_vectors, test_y))


# 构建待预测句子的向量
def get_predict_vectors(words):
    n_dim = 100
    model = Word2Vec.load(CUR_FILE + '/data/svm_data/w2v_model.pkl')
    # model.train(words)
    train_vectors = build_sentence_vector(words, n_dim, model)
    # print train_vectors.shape
    return train_vectors


# 对单个句子进行情感判断
def svm_predict(word_str):
    words = jieba.lcut(word_str)
    words_vectors = get_predict_vectors(words)
    clf = joblib.load(CUR_FILE + '/data/svm_data/model.pkl')

    result = clf.predict(words_vectors)
    return int(result[0])

    # if int(result[0]) == 1:
    #     print(word_str, ' 开发')
    # else:
    #     print(word_str, ' 非开发')


if __name__ == '__main__':
    print('---------------加载字典-----------------')
    jieba.load_userdict('./data/user_dict.txt')
    print('---------------文本数据预处理-----------------')
    X_train, X_test = load_file_and_preprocessing()
    print('---------------获取向量-----------------')
    get_train_vectors(X_train, X_test)
    print('---------------加载保存数据-----------------')
    train_vector, Y_train, test_vector, Y_test = get_data()
    print('---------------开始训练数据-----------------')
    svm_train(train_vector, Y_train, test_vector, Y_test)
    print('---------------开始预测-----------------')

    # 对输入句子情感进行判断
    # word='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # word='牛逼，好评'
    word = '前端'
    svm_predict(word)
