import math
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import  *

import sklearn
from sklearn.preprocessing import *
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.pipeline import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.feature_extraction import *
from sklearn.neighbors import *
from sklearn.feature_extraction.text import *
from sklearn.neural_network import *
from xgboost.sklearn import *
from lightgbm.sklearn import *

from pickle import *
from sklearn.externals.joblib import *

from keras.preprocessing.text import *
from keras.preprocessing.sequence import pad_sequences






def my_print_datas(train):
    set_option('display.width', 100)
    set_option('precision', 2)
    print(train.shape)
    print(train.head(5))
    print(train.dtypes)# if need transform object to value
    print(train.describe())
    print(train.info())  # if need handle missing data
    print(train.groupby('item_condition_id').size())  # if need standarize


def my_draw_datas(train):
    if MY_PLOT_SHOW:
        print(train.corr(method='pearson'))  # the bigger between features, the worse -1~+1
        print(train.skew())  # 0 is best,left or right base, if need standarize
        
        train.hist()  # if gaussian distribution
        plt.show()
        
        train.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
        plt.show()
        
        train.plot(kind='box', subplots=True, layout=(3, 3), sharex=False)
        plt.show()
        
        correlations = train.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, 9, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(train_column_names)
        ax.set_yticklabels(train_column_names)
        plt.show()
        
        scatter_matrix(train)
        plt.show()

def __2Dto1D(train,test):
    max_name_seq = np.max(
        [np.max(train.name.apply(lambda x: len(x))), np.max(test.name.apply(lambda x: len(x)))])
    max_seq_item_description = np.max([np.max(train.item_description.apply(lambda x: len(x)))
                                          , np.max(test.item_description.apply(lambda x: len(x)))])
    print(max_name_seq, max_seq_item_description)
    
    if MY_PLOT_SHOW:
        train.item_description.apply(lambda x: len(x)).hist()
        train.name.apply(lambda x: len(x)).hist()
    
    estimated_name_len=10
    estimated_item_des_len=75

    train_data_01 = pad_sequences(train.name,maxlen=estimated_name_len)
    test_data_01 = pad_sequences(test.name,maxlen=estimated_name_len)
    pca=PCA(n_components=estimated_name_len, copy=False)
    fit = pca.fit(train_data_01)
    print(fit.explained_variance_ratio_)#bigger, means can explain more other features by itself
    print(fit.components_)
    train_data_01 = fit.transform(train_data_01)
    test_data_01 = fit.transform(test_data_01)


    
    train_data_02 = pad_sequences(train.item_description,maxlen=estimated_item_des_len)
    test_data_02 = pad_sequences(test.item_description,maxlen=estimated_item_des_len)
    pca=PCA(n_components=estimated_item_des_len, copy=False)
    fit = pca.fit(train_data_02)
    print(fit.explained_variance_ratio_)#bigger, means can explain more other features by itself
    print(fit.components_)
    train_data_02 = fit.transform(train_data_02)
    test_data_02 = fit.transform(test_data_02)
    
    
    
    x_train=np.hstack([train_data_01,
                       train.item_condition_id.as_matrix().reshape(-1,1),
                       train.category_name.as_matrix().reshape(-1,1),
                       train.brand_name.as_matrix().reshape(-1,1),
                       train.shipping.as_matrix().reshape(-1,1),
                       train_data_02])
    y_train=train.price.as_matrix()
    x_test=np.hstack([test_data_01,
                      test.item_condition_id.as_matrix().reshape(-1,1),
                      test.category_name.as_matrix().reshape(-1,1),
                      test.brand_name.as_matrix().reshape(-1,1),
                      test.shipping.as_matrix().reshape(-1,1),
                      test_data_02])
    return x_train,y_train,x_test




def my_feature_extraction(df):
    df.brand_name.fillna('xiaofei', inplace=True)
    df.category_name.fillna('songwen', inplace=True)
    df.item_description.fillna('huihui', inplace=True)
    
    
    #brand_name
    pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
    df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "longwei"
    df["brand_name"] = df["brand_name"].astype("category")
    vect_brand = LabelBinarizer(sparse_output=True)
    X_brand = vect_brand.fit_transform(df["brand_name"])

    #category_name
    unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
    count_category = CountVectorizer()
    X_category = count_category.fit_transform(df["category_name"])

    #item_description
    count_descp = TfidfVectorizer(max_features=MAX_FEAT_DESCP,
                                  ngram_range=(1, 3),
                                  stop_words="english")
    X_descp = count_descp.fit_transform(df["item_description"])

    #item_condition_id, shipping
    df["item_condition_id"] = df["item_condition_id"].astype("category")
    X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
        "item_condition_id", "shipping"]], sparse=True).values)

    #name
    count = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = count.fit_transform(df["name"])

    X = scipy.sparse.hstack((X_dummies,
                             X_descp,
                             X_brand,
                             X_category,
                             X_name)).tocsr()

    
    return X




def my_preprocessing_data(X):
    '''
    #a&b can fit to same unit
    # a. MinMaxScaler(feature_range=(0,1))--------for g d t
    # b. StandardScaler(),scale()--------for gaussian input, LR, LR, LDA
    # c. Normalizer(copy=True,norm='l2'),normalize()----------for sparse feature, NN KNN. Most used in text classification, or cluster
    # d. Binarizer(copy=True,threshold=0.0)
    '''
    

    
    '''
    ss = StandardScaler()
    train.name =ss.fit_transform(train.name.reshape(-1,1))
    train.category_name = ss.fit_transform(train.category_name.reshape(-1,1))
    train.brand_name = ss.fit_transform(train.brand_name.reshape(-1,1))
    train.item_description = ss.fit_transform(train.item_description.reshape(-1,1))
    train.item_description_std = ss.fit_transform(train.item_description_std.reshape(-1,1))
    train.name_std = ss.fit_transform(train.name_std.reshape(-1,1))
    '''

    return X
    
def my_feature_selection(X,y):
    steps = []
    steps.append(('TSVD', TruncatedSVD(n_components=2000)))
    #steps.append(('KBest',SelectKBest(score_func=chi2, k=500)))
    combined_features = FeatureUnion(steps)
    X = combined_features.fit_transform(X,y)

    return X

def __rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(math.fabs(y_pred[i]) + 1) - math.log(math.fabs(y[i]) + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5
    


def my_normal_model_selection(X_train,y_train):

    # regression_piplies={}
    # regression_piplies['LR']=Pipeline([('LR',LinearRegression())])
    # regression_piplies['RIGE']=Pipeline([('RIGE',Ridge())])
    # regression_piplies['LA']=Pipeline([('LA',Lasso())])
    # regression_piplies['EN']=Pipeline([('EN',ElasticNet())])
    # regression_piplies['KN']=Pipeline([('KN',KNeighborsRegressor())])
    # regression_piplies['DT']=Pipeline([('DT',DecisionTreeRegressor())])
    # regression_piplies['SVM']=Pipeline([('SVM',SVR())])
    # results=[]
    #
    #
    # for key in regression_piplies:
    #     kf = KFold(n_splits=5,random_state=7)
    #     cv_result=cross_val_score(regression_piplies[key],X_train, y_train,cv=kf, n_jobs=-1)
    #     results.append(cv_result)
    #     print key, cv_result.mean(),cv_result.std()
    #
    # if MY_PLOT_SHOW:
    #     fig=plt.figure()
    #     fig.suptitle("Algorithm Comparison")
    #     ax=fig.add_subplot(111)
    #     plt.boxplot(results)
    #     ax.set_xticklabels(regression_piplies.keys())
    #     plt.show()

    #select the best non-ensemble model to do grid-search
    model=Lasso()
    parameters={'max_iter':[100,400,1000]}

    model = SVR()
    best_score, best_params = __my_grid_search(X_train,y_train,model,parameters)
    print(best_score, best_params)

    return model
    
    
def my_ensemble_model_selection(X_train,y_train):
    
    ensembles={}
    ensembles['BAG']=Pipeline([('BAG',BaggingRegressor())])
    ensembles['RF']=Pipeline([('RF',RandomForestRegressor())])
    ensembles['ET']=Pipeline([('ET',ExtraTreesRegressor())])

    ensembles['ADA']=Pipeline([('ADA',AdaBoostRegressor())])
    ensembles['GB']=Pipeline([('GB',GradientBoostingRegressor())])
    ensembles['XGB']=Pipeline([('XGB',XGBRegressor())])
    ensembles['GBM']=Pipeline([('GBM',LGBMRegressor())])

    
    results = []
    for key in ensembles:
        kf = KFold(n_splits=5, random_state=7)
        cv_result = cross_val_score(ensembles[key], X_train,y_train, cv=kf,n_jobs=-1)
        results.append(cv_result)
        print(key, cv_result.mean(), cv_result.std())

    if MY_PLOT_SHOW:
        fig = plt.figure()
        fig.suptitle("Algorithm Comparison")
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(ensembles.keys())
        plt.show()
    
    
    #select the best ensemble model to do grid-search
    model=XGBRegressor()
    parameters={'n_estimators':[10],'learning_rate':[0.1],'max_depth':[1],'booster':['gbtree'],'min_child_weight':[1],'subsample':[1.0],'random_state':[10]}
    best_score, best_params = __my_grid_search(X_train,y_train,model,parameters)
    print(best_score, best_params)


    return model

    
def __my_grid_search(x,y,model,parameters):
    #apply when parameters are less than 3
    kf = KFold(random_state=7, n_splits=5)#ShuffleSplit,KFold
    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=kf, n_jobs=-1)#n_jobs is important for time saving
    grid.fit(x,y)
    print(grid.get_params())
    
    return grid.best_score_, grid.best_params_

    
def my_draw_learning_curve(estimator,X,y,train_sizes=np.linspace(.05,1.,20)):
    if MY_PLOT_SHOW:
        train_size,train_score,test_score=learning_curve(estimator,X,y,train_sizes=train_sizes)
        train_score_mean=np.mean(train_score,axis=1)
        train_score_std=np.std(train_score,axis=1)
        test_score_mean=np.mean(test_score,axis=1)
        test_score_std=np.std(test_score,axis=1)
    
        plt.figure()
        plt.title('Learning Curve')
        plt.xlabel('Number of training set')
        plt.ylabel('Score')
        plt.grid()
    
        plt.fill_between(train_size,train_score_mean-train_score_std,train_score_mean+train_score_std,alpha=0.1,color='b')
        plt.fill_between(train_size,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.1,color='r')
        plt.plot(train_size,train_score_mean,'o-',color='b',label='Score in training set')
        plt.plot(train_size,test_score_mean,'o-',color='r',label='Score in cv set')
    
        plt.legend(loc='best')
        plt.show()
    
        midpoint = ((train_score_mean[-1]+train_score_std[-1]+test_score_mean[-1]-test_score_std[-1]))/2
        diff = (train_score_mean[-1]+train_score_std[-1])-(test_score_mean[-1]-test_score_std[-1])
        return midpoint,diff

MYROWS=100000
train_column_names = ['name', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping', 'item_description']
test_column_names = ['name', 'item_condition_id', 'category_name', 'brand_name', 'shipping','item_description']
label_column_names = ['price']
MY_PLOT_SHOW=True
NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

def PLOG(info):
    localtime = time.asctime(time.localtime(time.time()))
    print(info,':    ',localtime)
    
if __name__ =='__main__':
    
    #1. read data download data from https://www.kaggle.com/c/mercari-price-suggestion-challenge/data
    df_train = pd.read_table('./train.tsv',nrows=1000)
    df_test = pd.read_table('./test.tsv',nrows=1000)
    df = pd.concat([df_train, df_test], 0)
    nrow_train = df_train.shape[0]
    y_train = np.log1p(df_train["price"])
    
    #==================================Feature Engineering Start========================================================
    #2. understand data,can be called everywhere serveral times
    my_print_datas(df_train)
    print('my_print_datas')
    #3. watch data again, draw data,can be called everywhere serveral times
    my_draw_datas(df_train)
    
    #4. feature_extraction,fill_missing_data,one-hot,labelcoder,tokenizer,padsequence
    #all data to 1D numberic
    X=my_feature_extraction(df)
    print('my_feature_extraction')

    #5. preprocessing,standarize,scale,normalizer,minmaxselector
    X=my_preprocessing_data(X)
    
    #6. feature selection,K, feature_importance
    #X = my_feature_selection(X,y_train)
    # ==================================Feature Engineering End=========================================================

    X_train = X[:nrow_train]
    X_test = X[nrow_train:]
    print('Ridge')

    #7. normal model selection, pipelines, gridsearch, crossvalidate
    #model = my_normal_model_selection(X_train,y_train)
    
    #8. ensemble model selection, pipelines, gridsearch, crossvalidate
    #model = my_ensemble_model_selection(X_train,y_train)

    model = Ridge()
    #9. draw leanring curve
    my_draw_learning_curve(model,X_train,y_train)

    print('my_draw_learning_curve')

    

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(__rmsle(y_train, preds))


    df_test["price"] = np.expm1(preds)
    df_test[["test_id", "price"]].to_csv("submission.csv", index=False)

