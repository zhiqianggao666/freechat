import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import  *
from test import  *
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn
import sklearn.linear_model
from sklearn import cross_validation

# test_classifiers = ['NB','KNN', 'LR', 'RF', 'DT', 'SVM','LIBSVM', 'GBDT','ADA','MLP','XGBOOST']
test_classifiers = ['XGBOOST']

RATIO = 0.8



def show_data_in_figure(data):
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2,3),(0,0))
    data['Survived'].value_counts().plot(kind='bar')
    plt.title('alive status, 1 means alive')
    plt.ylabel('number')

    plt.subplot2grid((2,3),(0,1))
    data['Pclass'].value_counts().plot(kind='bar')
    plt.title('Pclass status')
    plt.ylabel('number')

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data['Survived'],data['Age'])
    plt.ylabel('Age')
    plt.grid(b=True,which='major',axis='y')
    plt.title('alive via age')

    plt.subplot2grid((2,3),(1,0),colspan=2)
    data.Age[data.Pclass==1].plot(kind='kde')
    data.Age[data.Pclass==2].plot(kind='kde')
    data.Age[data.Pclass==3].plot(kind='kde')
    plt.xlabel('Age')
    plt.ylabel('density')
    plt.title('Pclass via age')
    plt.legend(('1st,2nd,3rd'),loc='best')

    plt.subplot2grid((2,3),(1,2))
    data.Embarked.value_counts().plot(kind='bar')
    plt.title('Embarked status')
    plt.ylabel('number')



    plt.show()
    

    fig=plt.figure()
    fig.set(alpha=0.2)

    survived_0=data.Pclass[data.Survived==0].value_counts()
    survived_1=data.Pclass[data.Survived==1].value_counts()
    df=pandas.DataFrame({'alive':survived_1,'dead':survived_0})
    df.plot(kind='bar',stacked=True)
    plt.title('Pclass alive status')
    plt.xlabel('Pclass')
    plt.ylabel('number')
    plt.show()


    fig=plt.figure()
    fig.set(alpha=0.2)
    survived_m=data.Survived[data.Sex=='male'].value_counts()
    survived_f=data.Survived[data.Sex=='female'].value_counts()
    df=pandas.DataFrame({'male':survived_m,'female':survived_f})
    df.plot(kind='bar',stacked=True)
    plt.title('Sex alive status')
    plt.xlabel('Sex')
    plt.ylabel('number')
    plt.show()

    g=data.groupby(['SibSp','Survived'])
    df=pandas.DataFrame(g.count()['PassengerId'])

    g=data.groupby(['Parch','Survived'])
    df = pandas.DataFrame(g.count()['PassengerId'])

    print(data.Cabin.value_counts())


    fig=plt.figure()
    fig.set(alpha=0.2)

    survived_cabin=data.Survived[pandas.notnull(data.Cabin)].value_counts()
    survived_nocabin=data.Survived[pandas.isnull(data.Cabin)].value_counts()
    df=pandas.DataFrame({'have':survived_cabin,'not have':survived_nocabin}).transpose()
    df.plot(kind='bar',stacked=True)
    plt.title('Cabin status')
    plt.xlabel('have or not')
    plt.ylabel('number')
    plt.show()



def set_missing_ages(df):
    df.loc[df.Fare.isnull(),'Fare']=0
    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age=age_df[age_df.Age.notnull()].as_matrix()
    unknown_age=age_df[age_df.Age.isnull()].as_matrix()
    y=known_age[:,0]
    X=known_age[:,1:]

    rfr=RandomForestRegressor(random_state=0,n_estimators=200,n_jobs=-1)
    rfr.fit(X,y)

    predictedAges=rfr.predict(unknown_age[:,1::])
    df.loc[(df.Age.isnull()),'Age']=predictedAges

    return df,rfr

def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(),'Cabin']='Yes'
    df.loc[df.Cabin.isnull(),'Cabin']='No'
    return df

def generate_new_data(df):
    dummies_cabin=pandas.get_dummies(df.Cabin,prefix='Cabin')
    dummies_Embarked=pandas.get_dummies(df.Embarked,prefix='Embarked')
    dummies_Sex=pandas.get_dummies(df.Sex,prefix='Sex')
    dummies_Pclass=pandas.get_dummies(df.Pclass,prefix='Pclass')
    data=pandas.concat([df,dummies_cabin,dummies_Embarked,dummies_Sex,dummies_Pclass], axis=1)
    data.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
    return data

def scale_data(df):
    import sklearn.preprocessing as preprocessing
    #scaler = preprocessing.StandardScaler()
    #age_scale_param=scaler.fit([df.Age.as_matrix().reshape(-1,1)])
    #df.Age_scaled=scaler.fit_transform(df.Age,age_scale_param)
    #fare_scale_param=scaler.fit(df.Fare)
    #df.Fare_scaled=scaler.fit_transform(df.Fare,fare_scale_param)
    df.Age=preprocessing.scale(df.Age)
    df.Fare=preprocessing.scale(df.Fare)

    return df

def check_bad_cases():
    ori_data = pandas.read_csv('./train.csv', sep=',', header=0)

    split_train,split_cv=cross_validation.train_test_split(ori_data,test_size=0.3,random_state=0)
    st_df=split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')


    model=sklearn.linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-1)
    model.fit(st_df.as_matrix()[:,1:],st_df.as_matrix()[:,0])

    cv_df=split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = model.predict(cv_df.as_matrix()[:,1:])

    bad_cases= ori_data.loc[ori_data.PassengerId.isin(split_cv[predictions!=cv_df.as_matrix()[:,0]].PassengerId.values)]
    print(bad_cases)

def cross_validate(model,X,y):
    a= cross_validation.cross_val_score(model,X,y,cv=5)
    print(a)
    print(np.mean(a))

def check_model_parameter(train_df,model):
    infer1 = pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(model.coef_.T)})
    print(infer1)

def draw_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_size=np.linspace(.05,1.,20),verbose=0,plot=True):
    import numpy as np
    import  matplotlib.pyplot as plt
    from sklearn.learning_curve import learning_curve
    train_size,train_score,test_score=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_size,verbose=verbose)
    train_score_mean=np.mean(train_score,axis=1)
    train_score_std=np.std(train_score,axis=1)
    test_score_mean=np.mean(test_score,axis=1)
    test_score_std=np.std(test_score,axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('Number of training set')
        plt.ylabel('Score')
        #plt.gca().invert_yaxis()
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







if __name__ =='__main__':
    print('read data from csv file')
    data = pandas.read_csv('./train.csv', sep=',', header=0)
    #print pandas.__version__
    #print data.info()
    #print data.describe()
    # PassengerId NA
    # Survived LABEL    0/1
    # Pclass *          1/2/3
    # Name NA
    # Sex *              1/2
    # Age *              int
    # SibSp              int
    # Parch              int
    # Ticket NA
    # Fare *             float
    # Cabin NA
    # Embarked *         1/2/3
    #data = data.fillna(5)
    data,rfr=set_missing_ages(data)
    data=set_Cabin_type(data)
    show_data_in_figure(data)
    data = generate_new_data(data)
    scale_data(data)
    train_df = data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np=train_df.as_matrix()
    y=train_np[:,0]
    X=train_np[:,1:]


    M = len(y)
    pos = int(RATIO*M)
    train_x = X[:pos]
    train_y = y[:pos]
    print(len(train_y))
    test_x = X[pos:]
    test_y = y[pos:]
    print(len(test_y))
    model = None
    for i, name in enumerate(test_classifiers):
        model = do_training(name, train_x, train_y, test_x, test_y)

    #check_model_parameter(train_df,model)


    #cross_validate(model,X,y)
    #check_bad_cases()
    draw_learning_curve(model,'Learning Curve',X,y)

    test_data = pandas.read_csv('./test.csv', sep=',', header=0)

    test_data,rfr=set_missing_ages(test_data)
    test_data=set_Cabin_type(test_data)
    #show_data_in_figure(data)
    test_data = generate_new_data(test_data)
    scale_data(test_data)
    train_df = test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np=train_df.as_matrix()

    m = len(train_np)
    predict_y = np.ones(m)
    predict_x=train_np[:]

    for i, name in enumerate(test_classifiers):
       result = do_predicting(name,predict_x, predict_y)
       ans = test_data['PassengerId']
       ansD = ans.to_frame()
       other = pandas.DataFrame({'Survived':result})
       #print other
       ansD = ansD.join(other)
       #print ansD
       ansD.to_csv('./result.csv', columns=['PassengerId','Survived'], index = False)