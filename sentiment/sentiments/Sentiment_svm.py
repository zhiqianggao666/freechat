# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC

# 加载文件，导入数据,分词
def loadfile():
    neg=pd.read_excel('../data/neg.xls',header=None,index=None)
    pos=pd.read_excel('../data/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    #print pos['words']
    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    
    np.save('../data/svm_data/y_train.npy',y_train)
    np.save('../data/svm_data/y_test.npy',y_test)
    return x_train,x_test
 


#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
    
#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)
    
    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train, total_examples= len(x_train), epochs=100)
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    
    np.save('../data/svm_data/train_vecs.npy',train_vecs)
    print (train_vecs.shape)
    #Train word2vec on test tweets
    imdb_w2v.train(x_test, total_examples = len(x_test), epochs=100)
    imdb_w2v.save('../data/svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    np.save('../data/svm_data/test_vecs.npy',test_vecs)
    print(test_vecs.shape)



def get_data():
    train_vecs=np.load('../data/svm_data/train_vecs.npy')
    y_train=np.load('../data/svm_data/y_train.npy')
    test_vecs=np.load('../data/svm_data/test_vecs.npy')
    y_test=np.load('../data/svm_data/y_test.npy')
    return train_vecs,y_train,test_vecs,y_test
    

##训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, '../data/svm_data/svm_model/model.pkl')
    print(clf.score(test_vecs,y_test))
    
    
##得到待预测单个句子的词向量    
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('../data/svm_data/w2v_model/w2v_model.pkl')
    train_vecs = buildWordVector(words, n_dim,imdb_w2v)
    return train_vecs
    
####对单个句子进行情感判断    
def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('../data/svm_data/svm_model/model.pkl')
     
    result=clf.predict(words_vecs)
    
    if int(result[0])==1:
        print(string,' positive')
    else:
        print(string,' negative')
    
if __name__=='__main__':
    
    
    ##导入文件，处理保存为向量
    # x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
    # get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
    # train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
    # svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型

    #string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(string)
    