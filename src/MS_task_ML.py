# -*- coding: utf-8 -*-
"""
Created on Mon May 02 14:57:13 2016

@author: wux06
"""

#%% Import modules

# from scipy import io as spio
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import jj_basic_fn as JJ
#import time


#%% Define parameters

subject_id = 'NY607'

trlbeg_tm, trlend_tm = 0.0, 0.8
binsize = JJ.run_binsize


#%% Load in data, tranform to desired format

path_name = JJ.path_names[subject_id]
datafile_name = JJ.task_datafile_names[subject_id]

f = h5py.File(path_name+datafile_name, 'r')

trialidx = f['trialidx']
trialidx = np.array(trialidx)[0]
trialidx = trialidx.astype(int)

trialtms = f['trialtms']
trialtms = np.array(trialtms)[:,0]
RT = JJ.find_nearest(trialtms, trlbeg_tm)
trlbeg_idx = RT[0]
RT = JJ.find_nearest(trialtms, trlend_tm)
trlend_idx = RT[0]
trlsamp_num = np.arange(trlbeg_idx, trlend_idx).size
binnum = trlsamp_num // binsize

elec_names = JJ.load_elec_names(f)

binned_vec = np.empty(0)
for t in range(trialidx.size):
    exec("hgp_trl = f['hgp_trl%d']" % t)
    hgp_trl = np.array(hgp_trl)
    hgp_trl = hgp_trl[trlbeg_idx:trlend_idx, :]
    hgp_trl_binned = JJ.bin_2Dmat(hgp_trl, binnum, binsize, elec_names.size)
    if t==0:
        binned_vec = hgp_trl_binned
    else:
        binned_vec = np.concatenate((binned_vec, hgp_trl_binned), axis=0)
        if t%100 == 0:
            print "Imported and processed %d trials" % t
del hgp_trl, hgp_trl_binned

f.close()
print "Finished importing data"


#%% Select relevant data for machine learning

binned_vec = binned_vec[trialidx<10, :]
trialidx = trialidx[trialidx<10]
class_labels = np.unique(trialidx)
class_names = ['Sound', 'Face', 'Building', 'Rest']


#%% Machine learning

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

#from sklearn.preprocessing import StandardScaler

#from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
#from sklearn.multiclass import OneVsRestClassifier

#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression



classifier = 4 # 1:svm.SVC, 2:svm.LinearSVC, 3:OneVsRestClassifier, 4:svm.SVC+probability=True
               # 5:GaussianNaiveBayes, 6:LogisticRegression, 7:LinearDiscriminantAnalysis
do_scaling = 1
do_gridsearch = 1


defs = {}
defs['classifier'] = classifier
defs['do_scaling'] = do_scaling

test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(binned_vec, trialidx, test_size=test_size, 
                                                    random_state=42, stratify=trialidx)

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
CV = skf.split(np.zeros(len(y_train)), y_train)

if classifier <= 4:    
    kernel_name = 'linear' #'rbf'
    C_range = np.logspace(-5, -1, 17) #Loose search:(-8, 4, 13), Fine search:(-5, -1, 17)
    
    defs['kernel_name'] = kernel_name
    defs['C'] = 1.0 # default value
    defs['probability'] = False # default value    
    
    clf_try = JJ.clf_list(defs)
    tuned_params = dict(svc__C=C_range)
    if classifier==3:
        tuned_params = dict(estimator__svc__C=C_range)
                
elif classifier==5:
    clf_name = 'Gaussian Naive Bayes classifier'
    prr = np.ones(len(class_labels)) * (1.0/len(class_labels))
    defs['priors'] = prr

elif classifier==6:
    clf_name = 'Logistic Regression'
    defs['solver'] = 'lbfgs'
    defs['multi_class'] = 'multinomial'
    defs['max_iter'] = 200
    
elif classifier==7:
    clf_name = 'Linear Discriminant Analysis'
    prr = np.ones(len(class_labels)) * (1.0/len(class_labels))
    defs['solver'] = 'eigen'  # 'svd', 'lsqr', 'eigen'
    defs['shrinkage'] = 'auto'
    defs['priors'] = prr


if do_gridsearch:
    cv_scrs = np.zeros((len(C_range), n_fold))
    
    clf_grid = GridSearchCV(clf_try,
                            param_grid=tuned_params,
                            cv=CV,
                            verbose=3)
    clf_grid.fit(X_train, y_train)
    print('Best score: {}'.format(clf_grid.best_score_))
    print('Best parameters: {}'.format(clf_grid.best_params_))

    clf_best = clf_grid.best_estimator_

    ## This is unnecessary!
    ## clf_best is a pipeline object that also contains the fitted scaling step
    #if do_scaling:
    #    scaler = StandardScaler()
    #    X_train = scaler.fit_transform(X_train)
    #    X_test = scaler.transform(X_test)

    y_pred = clf_best.predict(X_test)

    for fd in np.arange(n_fold):
        exec("grid_scr = clf_grid.cv_results_['split%d_test_score']" % fd)
        cv_scrs[:, fd] = grid_scr
    
else:
    cv_scrs = []
        
    for train_index, test_index in CV:
        clf_try = JJ.clf_list(defs)
        clf_try.fit(X_train[train_index,:], y_train[train_index])
        cv_scrs.append(clf_try.score(X_train[test_index,:], y_train[test_index]))
    cv_scrs = np.array(cv_scrs)
    
    clf_try = JJ.clf_list(defs)
    clf_try.fit(X_train, y_train)
    y_pred = clf_try.predict(X_test)
        


heldout_scr = accuracy_score(y_test, y_pred)
confus_mat = confusion_matrix(y_test, y_pred, labels=class_labels)


# Plot confusion matrix
plt.figure(figsize=(8,8))
plt.imshow(confus_mat, interpolation='nearest', cmap=plt.cm.Blues)
ax = np.arange(len(class_labels))
plt.xticks(ax, class_names)
plt.yticks(ax, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
if classifier <= 4:
    title_text = 'Obtain best estimator via grid search\nValidate on held-out test-size={}, accuracy={:.4f}\n \
                  Plot confusion matrix\n{} MS\nSVM, {} kernel, C={}\n \
                  {}-fold cross-validation, best mean score={:.4f}\n'.format(
                  test_size, heldout_scr, 
                  subject_id, kernel_name, 
                  clf_grid.best_params_.values(), n_fold, clf_grid.best_score_)
elif classifier==5 or classifier==6 or classifier==7:
    title_text = '{}\nValidate on held-out test-size={}, accuracy={:.4f}\n \
                  Plot confusion matrix\n{} MS\n{}-fold cross-validation, mean score={:.4f}\n'.format(
                  clf_name, test_size, heldout_scr,
                  subject_id, n_fold, np.mean(cv_scrs))
plt.title(title_text)
axx, axy = np.meshgrid(ax, ax)
for x in np.ravel(axx):
    for y in np.ravel(axy):
        plt.text(x, y, str(confus_mat[y,x]),
                 fontsize=15, fontweight='normal', color='m',
                 horizontalalignment='center',
                 verticalalignment='center'
                 )
plt.tight_layout()

#joblib.dump(clf_grid, 'SVM_{}_grid.pkl'.format(kernel_name))
#clf_grid = joblib.load('SVM_{}_grid.pkl'.format(kernel_name))


#%% Train final model with all labeled data

#if do_scaling:
#    scaler = StandardScaler()
#    binned_vec = scaler.fit_transform(binned_vec)
#else:
#    scaler = []

if classifier <= 4:
    C_best = clf_grid.best_params_.values()[0]
    defs['C'] = C_best
      
if classifier==4:
    defs['probability'] = True

    
clf_final = JJ.clf_list(defs)
clf_final.fit(binned_vec, trialidx)

joblib.dump([classifier, do_scaling, clf_final], path_name+'ML_final.pkl', compress=3)


# Test predict_proba
if classifier==4:
    pred_prob = clf_final.predict_proba(binned_vec)
    max_idx = np.argmax(pred_prob, axis=1)
    y_pred = clf_final.classes_[max_idx]
    confus_mat_prob = confusion_matrix(trialidx, y_pred, labels=class_labels)
    
# LDA 'svd' or 'eigen' solver: plot transformed X
if classifier==7 and (defs['solver']=='svd' or defs['solver']=='eigen'):
    Xt = clf_final.transform(binned_vec)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'r', 'gold', 'saddlebrown']
    for i, c, n in zip(class_labels, colors, class_names):
        idx = np.where(trialidx==i)
        ax.scatter(Xt[idx,0], Xt[idx,1], Xt[idx,2], c=c, marker='o', label=n)
    
    ax.legend(loc=6, scatterpoints=1)
    ax.set_xlabel('LDA1')
    ax.set_ylabel('LDA2')
    ax.set_zlabel('LDA3')
    ax.set_title('Linear Discriminant Analysis Transformed Data')
    if True:
        with h5py.File(path_name+'LDA_TransX.h5', 'w') as f:
            f.create_dataset('Xt', data=Xt)
            f.create_dataset('trialidx', data=trialidx)
            f.create_dataset('class_labels', data=class_labels)
            f.create_dataset('class_names', data=class_names)
    





























