import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from matplotlib import cm as cm
import numpy as np
import math
import pandas as pd
from sklearn.metrics import roc_curve, auc
import six
import matplotlib
import os

import jj_basic_fn as JJ
import pandas as pd
import numpy as np
from hyperparams import Hyperparams as hp
import prep
import modules


def plot_epoch_mean(patient_list, if_save = 0, label = ''):
    #sample label : '_weekly'
    for patient in patient_list:
        patient.print_features_property()
        ptid = patient.id
        dat = patient.daily
        dat_epi_agg, dat_le_agg, dat_epi_agg_ste, dat_le_agg_ste = prep.dat_agg(dat)
        if ptid == '229w':
            if_bar = 1
            xlabel = 'epoch(week)'
        else:
            if_bar = 0
            xlabel = 'epoch(month)'
        if if_bar:
            epoch_label_dict = patient.epoch_label_dict
            type(epoch_label_dict)
            colors = [] #colors for plt.bar
            for key, val in epoch_label_dict.items():
                colors.append('red' if val else 'blue')
            plt.figure()
            #plt.bar(range(dat_le_agg.shape[0] - 1),np.array(dat_le_agg.iloc[:-1]), color = colors)
            plt.bar(range(dat_le_agg.shape[0]),np.array(dat_le_agg), color = colors)
            plt.xlabel('weeks', fontsize=hp.label_fontsize)
            plt.ylabel('mean long episode count per day', fontsize=hp.label_fontsize)
            plt.title('Patient {0}'.format(ptid), fontsize=hp.label_fontsize)
            if if_save:
                plt.savefig('../fig/'+ ptid + '/' + 'mean_long_episode_count' + '.png')
            plt.show()
        else:
            epoch_label_dict = patient.epoch_label_dict
            colors = [] #colors for plt.bar
            good_idx = []
            bad_idx = []
            for key, val in epoch_label_dict.items():
                # colors.append('green' if val else 'red')
                if val:
                    good_idx.append(key)
                else:
                    bad_idx.append(key)

            fig, ax = plt.subplots(1,1)
            ax.set_xticks(range(dat_le_agg.shape[0]))
            ax.set_xticklabels(range(1,dat_le_agg.shape[0] + 1))
            # plt.plot(dat_epi_agg, label = 'episodes start mean')
            # plt.plot(dat_epi_agg + dat_epi_agg_ste,linestyle='dashed', label = 'episodes start mean + sem')
            # plt.plot(dat_epi_agg - dat_epi_agg_ste,linestyle='dashed', label = 'episodes start mean - sem')
            #plt.errorbar(range(dat_le_agg.iloc[good_idx].shape[0]), np.array(dat_le_agg.iloc[good_idx]),yerr=np.array(dat_le_agg_ste.iloc[good_idx]), fmt='o', mfc='red')
            plt.errorbar(np.array(good_idx), np.array(dat_le_agg.iloc[good_idx]),yerr=np.array(dat_le_agg_ste.iloc[good_idx]), fmt='o', mfc='blue',ecolor='black')
            plt.errorbar(np.array(bad_idx), np.array(dat_le_agg.iloc[bad_idx]),yerr=np.array(dat_le_agg_ste.iloc[bad_idx]), fmt='o', mfc='red',ecolor='black')
            plt.plot(dat_le_agg, label = 'long episode mean', color = 'black')
            #plt.plot(dat_le_agg, label = 'long episode mean', color = 'black',marker='o', markerfacecolor = 'r')
            #plt.plot(dat_le_agg + dat_le_agg_ste,linestyle='dashed', label = 'long episode mean + sem', color = 'black')
            #plt.plot(dat_le_agg - dat_le_agg_ste,linestyle='dashed', label = 'long episode mean - sem', color = 'black')
            #plt.title('Patient {0}: period {1} - {2}'.format(ptid, period_start, period_end))
            plt.title('Patient {0}'.format(ptid), fontsize=hp.label_fontsize)
            plt.xlabel(xlabel, fontsize=hp.label_fontsize)
            plt.ylabel('mean long episode count per day', fontsize=hp.label_fontsize)
            plt.tight_layout()
            if if_save:
                directory = '../fig/mean_long_episode_count'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(directory + '/' + patient.id + label + 'mean_long_episode_count' + '.png')
            plt.show()


def plot_confusion_matrix(cm, classes = np.array(['good', 'bad']),
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues, if_save = 0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt

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
      plt.savefig('../fig/'+ figure_title + '.png')


def show_confusion_matrix(y_test, y_pred, clf_name, if_save = 0):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, title = clf_name, if_save = if_save) 


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


def plot_roc_all(X_test, y_test, pat, if_save = 0):
    classifier_list = [1,2,5,6,7]
    cmap = get_cmap(len(classifier_list))
    lw = 2
    plt.figure()
    ax = plt.subplot(111)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    for i,classifier_int in enumerate(classifier_list):

        y_score, accuracy, y_pred, clf_name = JJ.load_score(classifier_int, X_test, y_test, pat)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)        
        
        plt.plot(fpr, tpr, color=cmap(i),
               lw=lw, label='%s (AUC = %0.2f)' % (clf_name, roc_auc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=hp.label_fontsize)
        plt.ylabel('True Positive Rate', fontsize=hp.label_fontsize)
        plt.title('Patient ' + pat.id + ': ROC for all classifiers', fontsize=hp.label_fontsize)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.0 , box.width, box.height * 1])
    # ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
    #           shadow = True, ncol = 4, prop = {'size':10})
    plt.legend(prop={'size': 14})
    if if_save:
        plt.savefig('../fig/'+ pat.id + '/' + 'roc_all' + '.png')
    plt.show()


def correlation_matrix(df):
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(16,16))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    plt.title('Abalone Feature Correlation')
    labels=[]
    for x in df.columns.tolist():
        labels.append(x)
    labels = labels[:-1]
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels,fontsize=6, rotation = 30)
    ax1.set_yticklabels(labels,fontsize=6, rotation = 30)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap('Spectral', n)

def feature_importance(pat, classifier_int, if_save = 0, if_abs = 1):
    
    int2name = hp.int2name
    clf_name = int2name[classifier_int]
    clf = pat.estimator[classifier_int]
    classifier_type1 = [1]
    classifier_type2 = [6,7]
    topk = 3 #print topk important features
    if classifier_int in classifier_type1:
        
            coef = clf.coef_.reshape(6,4)
    elif classifier_int in classifier_type2:        
            coef = clf.feature_importances_.reshape(6,4)

    if if_abs == 1:
        coef = np.abs(coef)
        cmap = plt.cm.Blues
    else:
        cmap = plt.cm.Reds
    df = pd.DataFrame(coef, index = hp.powerbands1, columns = hp.channel)
    
    import seaborn as sns
    fig = plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    r = sns.heatmap(coef, cmap=cmap)
    label = 'patient '+ pat.id
    r.set_title("Feature importance heatmap of {} for {}".format(clf_name, label), fontsize=hp.label_fontsize -2)
    ax.set_yticklabels(df.index, fontsize=hp.label_fontsize-8, verticalalignment = 'center')
    ax.set_xticklabels(df.columns, fontsize=hp.label_fontsize-2)
    plt.show()
    inds = np.argpartition(coef.ravel(), -topk)[-topk:]
    feature_names = []
    for ind in inds:
        feature_name = ', ' + hp.powerbands[ind // 4 + 1] + ' ' + hp.channel[ind % 4] + ' '
        feature_names.append(feature_name)
    print('The 3 most important features for ' + str(clf_name) + ' are' + feature_names[0] + feature_names[1] + feature_names[2])
    print(coef)


    if if_save:
        plt.savefig('../fig/'+ pat.id + '/fi_' + clf_name + '.png')


def render_mpl_table(data, pat, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, label = None, if_save = 0, **kwargs):
    #plt.figure()
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    if label != None:
        plt.title(label, fontsize=hp.label_fontsize)
    if if_save:
        plt.savefig('../fig/'+ pat.id + '/' + label + '.png')
    plt.show()
    return 

def get_scatter_plot_data(pat, drop_list = [], if_remove_sleep = 1, if_remove_icd = 1):
    dlist = ['region_start_time', 'epoch', 'filename', 'if_stimulated', 'id']
    dlist.extend(drop_list)
    dat0 = pat.features
    dat = modules.remove_outliers(dat0)
    if if_remove_sleep:
        if 'sleep' in dat.columns:
            dlist.append('sleep')
    if if_remove_icd:
        dlist.append('i12')
        dlist.append('i34')
    
    X = dat.drop(dlist, axis = 1, inplace = False)
    #X = add_label_sti(X)
    return X
