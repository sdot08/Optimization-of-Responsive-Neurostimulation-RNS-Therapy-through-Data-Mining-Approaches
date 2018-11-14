# -*- coding: utf-8 -*-

import numpy as np
import math
import pandas as pd
import logging
import sys
import matplotlib.pyplot as plt
import time
from random import shuffle
import itertools
import pickle
from numpy.linalg import inv
import operator

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, make_scorer, roc_curve, auc
from sklearn.model_selection import train_test_split


import plot_funcs 
from hyperparams import Hyperparams as hp
from patient import patient
import prep

    
    
def clf_list(defs):
    classifier = defs['classifier']
    if classifier == 1:
        clf = LogisticRegression(max_iter = defs['max_iter'], class_weight = defs['class_weight'])
    
    elif classifier==2:
        clf = SVC(class_weight = defs['class_weight'])

    elif classifier == 3:
        clf = GaussianNB(priors=defs['priors'])
    elif classifier == 4:
        clf = LinearDiscriminantAnalysis(solver=defs['solver'], shrinkage=defs['shrinkage'], priors=defs['priors'])
        

    elif classifier==5:
      clf = DecisionTreeClassifier(class_weight='balanced')
    elif classifier == 6:
      clf = RandomForestClassifier(n_estimators=defs['n_estimators'], class_weight='balanced')   
    elif classifier == 7:
      clf = ensemble.GradientBoostingClassifier(n_estimators=defs['n_estimators'])
    
    return clf


  
def show_result(y_pred, y_test, df, clf_name = '', if_save = 0):
    heldout_scr = accuracy_score(y_test, y_pred)
    plot_funcs.show_confusion_matrix(y_test, y_pred, clf_name, if_save)
    cols_to_keep = ['params', 'mean_test_score']
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = pd.DataFrame(df_toshow.sort_values(by=["mean_test_score"],  ascending=False))
    display(pd.DataFrame(df_toshow))
    return df_toshow


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)







def scores_estimators(X_test, y_test, pat, if_save = 0):
    int2name = hp.int2name
    n_estimator = hp.num_classifier
    auc_dict = {}
    acc_dict = {}
    estimators = [1,2,5,6,7]
    for i in estimators:
        y_score, accuracy,_ , name = load_score(i, X_test, y_test, pat)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        #auc = pickle.load(open(prepath + 'Best_score_for_' + str(name) + '.p', "rb" ))
        auc_dict[name] = roc_auc
        acc_dict[name] = accuracy
        
    sorted_auc_dict = sorted(auc_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_acc_dict = sorted(acc_dict.items(), key=operator.itemgetter(1), reverse=True)
    plot_funcs.render_mpl_table(pd.DataFrame(sorted_auc_dict, columns = ['Classifier', 'AUC']), pat = pat, label = 'AUC for patient {}'.format(pat.id), if_save = if_save)
    plot_funcs.render_mpl_table(pd.DataFrame(sorted_acc_dict, columns = ['Classifier', 'Accuracy']), pat = pat, label = 'Accuracy for patient {}'.format(pat.id), if_save = if_save)

    #display(pd.DataFrame(sorted_auc_dict, columns = ['Classifier', 'AUC']))
    # display(pd.DataFrame(sorted_acc_dict, columns = ['Classifier', 'Accuracy']))


def load_score(classifier_int, X_test, y_test, pat):
    int2name = hp.int2name
    clf_name = int2name[classifier_int]
    clf = pat.estimator[classifier_int]
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    if classifier_int == 3 or classifier_int == 5 or classifier_int == 6:
        y_prob = clf.predict_proba(X_test)[:,1]
        y_prob = [prob + 0.001 * (prob < 0.001) - 0.001 * (1 - prob < 0.001) for prob in y_prob]
        y_score = np.array([-np.log(1/prob - 1) for prob in y_prob])
    else:
        y_score = clf.decision_function(X_test)
    return y_score, accuracy, y_pred, clf_name


def estimator_performance(classifier_int, X_test, y_test, pat, if_plot_c = 0, if_plot_roc = 0, plot_all = 0):
    y_score, accuracy, y_pred, clf_name = load_score(classifier_int, X_test, y_test, pat)
    if if_plot_c:
        plot_funcs.show_confusion_matrix(y_test, y_pred, clf_name)
    if if_plot_roc:
        plot_funcs.plot_roc(y_score, y_test)
    return

from sklearn.ensemble import VotingClassifier
def ensemble_model(X_train,y_train,X_test, y_test, pat, if_save = 0):
    int2name = {1:'Logistic Regression', 2: 'SVM', 3: 'Gaussian Naive Bayes classifier', 4:'Linear Discriminant Analysis', 5:'decision tree', 6:'random forest', 7:'gradient boosting'}
    classifier_list = [1,2,6,7]
    estimators =  []
    for classifier_int in classifier_list:
      clf_name = int2name[classifier_int]
      clf = pat.estimator[classifier_int]
      estimators.append((clf_name, clf))
    eclf = VotingClassifier(estimators=
            estimators, voting='hard')
    eclf.fit(X_train, y_train)
    clf_name = 'ensemble'

    y_pred = eclf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)




def select_data(dat, select_dict = None, keep_list = None):
    data = dat.copy()
    if keep_list:
      data = data.loc[:, keep_list]
    if select_dict:
      for key in select_dict:
          val = select_dict[key]
          data = data.loc[data[key] == val]
    return data



def get_variable_name(namelist):
    output = []
    for item in namelist:
        for i in range(1,5):
            output.append(item + str(i))
    return output

def scatter_plot_3d(data, patid,var_list):
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
     
    # Dataset

    var1, var2, var3 = var_list[0], var_list[1], var_list[2]
    keep_list = [var1,var2,var3,'patid', 'label']
    dfTure = select_data(data,select_dict = {'patid':patid, 'label':True}, keep_list = keep_list) 
    dfFalse = select_data(data,select_dict = {'patid':patid, 'label':False}, keep_list = keep_list) 

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dfTure[var1], dfTure[var2], dfTure[var3], c='skyblue', s=10)
    ax.scatter(dfFalse[var1], dfFalse[var2], dfFalse[var3], c='r', s=10)

    ax.view_init(30, 185)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel(var3)
    plt.show()
 
def add_label_sti(dat):
    output = dat.copy()
    lab1 = 'label'
    lab2 = 'if_stimulated'
    lab3 = 'label_sti'
    output.loc[(dat[lab1] == True) & (dat[lab2] == True), lab3] = 'Good&Sti'
    output.loc[(dat[lab1] == True) & (dat[lab2] == False), lab3] = 'Good&NoSti'
    output.loc[(dat[lab1] == False) & (dat[lab2] == True), lab3] = 'Bad&Sti'
    output.loc[(dat[lab1] == False) & (dat[lab2] == False), lab3] = 'Bad&NoSti'

    return output





def df_str2date(dat,col):
    dat.loc[:,col] = pd.to_datetime(dat.loc[:,col])
    return dat


