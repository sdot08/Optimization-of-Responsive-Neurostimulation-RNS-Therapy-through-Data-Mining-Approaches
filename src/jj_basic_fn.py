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

#matrix inverse

#default size of the graph
plt.rcParams['figure.figsize'] = (10.0, 8.0) 

Fs = 512

run_binsize_sec = 0.08 # Try1:0.02, Try2:0.08
run_binsize = int(math.floor(run_binsize_sec * Fs)) 

run_binnum = {"NY451":10,  # Try1:41
              "NY455":10,
              "NY607":10} 

path_names = {"NY451":'/space/mdeh5/1/halgdev/projects/pdelprato/Results/MS/NY451_XJ/XJ-V2_GSsmooth10ms_cfgfiltord2/Machine_learning/',
              "NY455":'/space/mdeh5/1/halgdev/projects/pdelprato/Results/MS/NY455/XJ-V2_GSsmooth10ms_cfgfiltord2/Machine_learning/',
              "NY607":'/space/mdeh5/1/halgdev/projects/pdelprato/Results/MS/NY607/Machine_learning/'}

task_datafile_names = {"NY451":'hgp_trials_final_py_NoDemean.mat',  #'hgp_trials_final_py.mat'
                       "NY455":'hgp_trials_final_py.mat',
                       "NY607":'hgp_trials_final_py_NoDemean.mat'} 

prerest_datafile_names = {"NY451":'hgp_Day3_NightSleep_py_PCAcln.mat',
                          "NY455":'hgp_Day2_NightSleep_py_PCAcln.mat',
                          "NY607":'hgp_prerest_py_PCAcln.mat'}

postrest_datafile_names = {"NY451":'hgp_Day4_NightSleep_py_PCAcln.mat',
                           "NY455":'hgp_Day4_NightSleep_py_PCAcln.mat',
                           "NY607":'hgp_postrest_py_PCAcln.mat'}

clffile_names = {"NY451":'NY451_MS_LDA_classifier7_eigen_final.pkl',  #'NY451_MS_SVM_classifier4_final.pkl'
                 "NY455":'NY455_MS_SVM_classifier4_final.pkl',
                 "NY607":'NY607_MS_LDA_classifier7_eigen_final.pkl'} #'NY607_MS_SVM_classifier4_final.pkl'

fullscreen_size = (20,11)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def load_elec_names(f):
    elec_names = []
    for e in range(f['elec_names'].size):
        exec("elecn = f['elec_names%d']" % e)
        elecn = np.array(elecn)
        txt = ''
        for subarray in elecn:
            for strnum in subarray:
                txt += unichr(strnum)
        elec_names.append(txt)
    elec_names = np.array(elec_names)
    return elec_names


def bin_2Dmat(mat, binnum, binsize, rownum):
    mat = mat[:(binnum*binsize), :]
    mat = mat.reshape(binnum, binsize, rownum)
    mat_binned = mat.mean(axis=1)
    mat_binned = np.ravel(mat_binned)[np.newaxis,:]
    return mat_binned
    
    
def clf_list(defs):
    classifier = defs['classifier']
    if classifier == 1:
        clf = LogisticRegression(max_iter = defs['max_iter'], class_weight = defs['class_weight'])
    
    elif classifier==2:
        clf = SVC()

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
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()

  





    