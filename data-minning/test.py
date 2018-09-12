#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import _pickle as pickle
import sklearn.externals.joblib as jl
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict, svm_save_model, svm_read_problem, svm_load_model
from sklearn.model_selection import  *



# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier2(train_x, train_y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier3(train_x, train_y):
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    model.fit(train_x, train_y)
    return model

# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=300)
    # model = RandomForestClassifier()
    model.fit(train_x, train_y)

    # n_estimators=[300,350,400,450,500]
    #
    # param_grid = dict(n_estimators=n_estimators)
    # kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
    # grid_search = GridSearchCV(model,param_grid,scoring='neg_log_loss',n_jobs=-1,cv=kfold)
    # grid_result = grid_search.fit(np.array(train_x), np.array(train_y))
    # print grid_result.best_score_,'***********',grid_result.best_params_


    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model

def bagging_classifier(train_x, train_y):
    from sklearn.ensemble import BaggingClassifier
    model = BaggingClassifier(base_estimator=None)
    model.fit(train_x, train_y)
    return model


def voting_classifier(train_x, train_y):
    from sklearn.ensemble import VotingClassifier
    estimators = []
    from sklearn import tree
    model1 = tree.DecisionTreeClassifier()
    from sklearn.linear_model import LogisticRegression
    model2 = LogisticRegression(penalty='l2')
    estimators.append(['dt',model1])
    estimators.append(['lr', model2])
    
    estimators.append('lr','')
    model = VotingClassifier(estimators=estimators)
    model.fit(train_x, train_y)
    return model

# Ada
def ada_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(base_estimator=None,n_estimators=300)
    model.fit(train_x, train_y)
    # n_estimators = [1,10,30,40,50,100,200]
    # param_grid = dict(n_estimators=n_estimators)
    # kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    # grid_search = GridSearchCV(model,param_grid,scoring='neg_log_loss',n_jobs=-1,cv=kfold)
    # grid_result = grid_search.fit(np.array(train_x), np.array(train_y))
    # print grid_result.best_score_,'***********',grid_result.best_params_
    return model

# MLP
def mlp_classifier(train_x, train_y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=200)
    #model.fit(train_x, train_y)
    #hidden_layer_sizes = [100,200, 300, 400]
    #param_grid = dict(hidden_layer_sizes=hidden_layer_sizes)
    #kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
    #grid_search = GridSearchCV(model,param_grid,scoring='neg_log_loss',n_jobs=-1,cv=kfold)
    #grid_result = grid_search.fit(np.array(train_x), np.array(train_y))
    #print grid_result.best_score_,'***********',grid_result.best_params_
    from sklearn.ensemble import  BaggingClassifier
    bagging_model = BaggingClassifier(model,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
    bagging_model.fit(train_x, train_y)
    return bagging_model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='poly', probability=True)
    model.fit(train_x, train_y)
    #scores = cross_val_score(model,train_x,train_y,cv=10,scoring='accuracy')

    return model

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='poly', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='poly', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def xgboost_classifier1(train_x, train_y):
    from xgboost.sklearn import XGBClassifier
    model = XGBClassifier()
    # model = XGBClassifier(silent=1,
    #                       learning_rate=0.3,
    #                       n_estimators=100,
    #                       max_depth=6,
    #                       min_child_weight=1,
    #                       gamma=0,
    #                       subsample=1,
    #                       colsample_bytree=1,
    #                       objective='binary:logistic',
    #                       nthread=4,
    #                       scale_pos_weight=1,
    #                       seed=1000)
    model.fit(np.array(train_x), np.array(train_y))

    # from xgboost import plot_importance
    # from matplotlib import pyplot
    # plot_importance(model)
    # pyplot.show()

    return model

def xgboost_classifier(train_x, train_y):
    from xgboost.sklearn import XGBClassifier
    # model = XGBClassifier()
    model = XGBClassifier(silent=1,
                          learning_rate=0.1,
                          n_estimators=60,
                          max_depth=6,
                          min_child_weight=0.4,
                          gamma=0.5,
                          subsample=0.4,
                          colsample_bytree=1,
                          objective='binary:logistic',
                          nthread=4,
                          scale_pos_weight=1,
                          seed=1000)
    #max_depth=[2,3,4,5,6,7]
    #learning_rate = [0.01,0.05,0.1,0.2,0.4,0.8,1]
    #n_estimators = [30,60, 80, 100, 150, 200]
    #param_grid = dict()
    #kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
    #grid_search = GridSearchCV(model,param_grid,scoring='neg_log_loss',n_jobs=-1,cv=kfold)
    #grid_result = grid_search.fit(np.array(train_x), np.array(train_y))
    #print grid_result.best_score_,'***********',grid_result.best_params_



    model.fit(train_x, train_y)
    #
    # from xgboost import plot_importance
    # from matplotlib import pyplot
    # plot_importance(model)
    # pyplot.show()

    return model

def do_training(classifier_name,train_x,train_y,test_x,test_y):
    model_save_file = str('./models/')+classifier_name+str('.model')
    if classifier_name == 'LIBSVM':
        prob = svm_problem(np.array(train_y).tolist(), np.array(train_x).tolist())
        param = svm_parameter('-s 1 -t 1 -q -d 3')
        # param = svm_parameter('-t 2 -q')
        model = svm_train(prob, param)
        svm_save_model('./models/{}.model'.format(classifier_name), model)
        svm_predict(np.array(test_y).tolist(), np.array(test_x).tolist(), model)
        return model

    model_save = {}
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier,
                   'ADA':ada_boosting_classifier,
                   'MLP': mlp_classifier,
                   'XGBOOST': xgboost_classifier
                   }
    model = classifiers[classifier_name](train_x, train_y)
    model_save[classifier_name]=model
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
    jl.dump(model_save, model_save_file)
    return model

def drawline(x,y1,y2,y3,y4,title):
    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    plt.title(title)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.plot(x,y1)

    plt.subplot(2,2,2)
    plt.title(title)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.plot(x,y2)

    plt.subplot(2,2,3)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.plot(x,y3)

    plt.subplot(2,2,4)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.plot(x,y4)

    plt.show()

def do_predicting(classifier_name,test_x,test_y):
    model_save_file = str('./models/')+classifier_name+str('.model')
    if classifier_name == 'LIBSVM':
        model = svm_load_model('./models/{}.model'.format(classifier_name))
        p_labels, p_acc, p_vals = svm_predict(test_y, np.array(test_x).tolist(), model)
        return p_labels

    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier,
                   'ADA':ada_boosting_classifier,
                   'MLP': mlp_classifier
                   }
    model = jl.load(model_save_file)[classifier_name]
    predict = model.predict(test_x)
    #accuracy = metrics.accuracy_score(test_y, predict)
    #print 'accuracy: %.2f%%' % (100 * accuracy)
    return predict