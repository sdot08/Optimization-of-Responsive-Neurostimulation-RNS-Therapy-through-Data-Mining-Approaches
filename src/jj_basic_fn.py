# -*- coding: utf-8 -*-
"""
Created on Mon May 02 19:54:45 2016

@author: wux06
"""

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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
from sklearn.metrics import mean_squared_error, make_scorer, roc_curve, auc
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
from sklearn.externals import joblib
from sklearn import ensemble
#default size of the graph
plt.rcParams['figure.figsize'] = (10.0, 8.0) 
    
    
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


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes = np.array(['good', 'bad']),
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues, if_save = 0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    figure_title = 'Confusion matrix for ' + str(title)
    plt.title(figure_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if if_save:
      plt.savefig('../fig/'+ figure_title + '.jpg')

def show_confusion_matrix(y_test, y_pred, clf_name, if_save = 0):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, title = clf_name, if_save = if_save)    

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Compute ROC curve and ROC area for each class
def plot_roc(y_score, y_test, n_classes = 2):
  fpr, tpr, _ = roc_curve(y_test, y_score)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap('Spectral', n)

def load_score(classifier_int, X_test, y_test):
    int2name = {1:'Logistic Regression', 2: 'SVM', 3: 'Gaussian Naive Bayes classifier', 4:'Linear Discriminant Analysis', 5:'decision tree', 6:'random forest', 7:'gradient boosting'}
    clf_name = int2name[classifier_int]
    clf = pickle.load(open('best_estimator_for_' + str(clf_name) + '.p', "rb" ))
    score = pickle.load(open('Best_score_for_' + str(clf_name) + '.p', "rb" ))
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    if classifier_int == 3 or classifier_int == 5 or classifier_int == 6:
        y_prob = clf.predict_proba(X_test)[:,1]
        y_prob = [prob + 0.001 * (prob < 0.001) - 0.001 * (1 - prob < 0.001) for prob in y_prob]
        y_score = np.array([-np.log(1/prob - 1) for prob in y_prob])
    else:
        y_score = clf.decision_function(X_test)
    return y_score, accuracy, y_pred, clf_name

def estimator_performance(classifier_int, X_test, y_test, if_plot_c = 0, if_plot_roc = 0, plot_all = 0):
    y_score, accuracy, y_pred, clf_name = load_score(classifier_int, X_test, y_test, )
    if if_plot_c:
        show_confusion_matrix(y_test, y_pred, clf_name)
    if if_plot_roc:
        plot_roc(y_score, y_test)
    return

def scores_estimators(X_test, y_test):
    int2name = {1:'Logistic Regression', 2: 'SVM', 3: 'Gaussian Naive Bayes classifier', 4:'Linear Discriminant Analysis', 5:'decision tree', 6:'random forest', 7:'gradient boosting'}
    n_estimator = 7
    auc_dict = {}
    acc_dict = {}
    estimators = [1,2,5,6,7]
    
    for i in estimators:
        _, accuracy,_ , name = load_score(i, X_test, y_test)
        auc = pickle.load(open('Best_score_for_' + str(name) + '.p', "rb" ))
        auc_dict[name] = auc
        acc_dict[name] = accuracy
    print(auc_dict)
    sorted_auc_dict = sorted(auc_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_acc_dict = sorted(acc_dict.items(), key=operator.itemgetter(1), reverse=True)
    display(pd.DataFrame(sorted_auc_dict, columns = ['Classifier', 'AUC']))
    display(pd.DataFrame(sorted_acc_dict, columns = ['Classifier', 'Accuracy']))

from sklearn.metrics import mean_squared_error, make_scorer, roc_curve, auc

def plot_roc_all(X_test, y_test):
    classifier_list = [1,2,5,6,7]
    cmap = get_cmap(len(classifier_list))
    lw = 2
    plt.figure()
    ax = plt.subplot(111)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    for i,classifier_int in enumerate(classifier_list):

        y_score, accuracy, y_pred, clf_name = load_score(classifier_int, X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)        
        
        plt.plot(fpr, tpr, color=cmap(i),
               lw=lw, label='%s (AUC = %0.2f)' % (clf_name, roc_auc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.0 , box.width, box.height * 1])
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
              shadow = True, ncol = 4, prop = {'size':10})
    plt.show()

def show_result(y_pred, y_test, df, clf_name = '', if_save = 0):
    heldout_scr = accuracy_score(y_test, y_pred)
    show_confusion_matrix(y_test, y_pred, clf_name, if_save)
    cols_to_keep = ['params', 'mean_test_score']
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = pd.DataFrame(df_toshow.sort_values(by=["mean_test_score"],  ascending=False))
    display(pd.DataFrame(df_toshow))
    return df_toshow

from sklearn.ensemble import VotingClassifier
def ensemble_model(X_train,y_train,X_test, y_test,if_save = 0):
    int2name = {1:'Logistic Regression', 2: 'SVM', 3: 'Gaussian Naive Bayes classifier', 4:'Linear Discriminant Analysis', 5:'decision tree', 6:'random forest', 7:'gradient boosting'}
    classifier_list = [1,2,6,7]
    estimators =  []
    for classifier_int in classifier_list:
      clf_name = int2name[classifier_int]
      clf = pickle.load(open('best_estimator_for_' + str(clf_name) + '.p', "rb" ))
      estimators.append((clf_name, clf))
    eclf = VotingClassifier(estimators=
            estimators, voting='hard')
    eclf.fit(X_train, y_train)
    clf_name = 'ensemble'

    y_pred = eclf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    if if_save:
        save_object(eclf, 'best_estimator_for_' + str(clf_name) + '.p')
        save_object(accuracy, 'Best_score_for_' + str(clf_name) + '.p')
