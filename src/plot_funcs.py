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


import jj_basic_fn as JJ
import pandas as pd
import numpy as np
from hyperparams import Hyperparams as hp
import prep

def plot_epoch_mean(patient_list):
    for patient in patient_list:
        ptid = patient.id
        dat = patient.daily
        dat_epi_agg, dat_le_agg, dat_epi_agg_ste, dat_le_agg_ste = prep.dat_agg(dat)
        plt.figure()
        fig, ax = plt.subplots(1,1)
        ax.set_xticks(range(dat_le_agg.shape[0]))
        ax.set_xticklabels(range(1,dat_le_agg.shape[0] + 1))
        plt.plot(dat_epi_agg, label = 'episodes start mean')
        plt.plot(dat_epi_agg + dat_epi_agg_ste,linestyle='dashed', label = 'episodes start mean + sem')
        plt.plot(dat_epi_agg - dat_epi_agg_ste,linestyle='dashed', label = 'episodes start mean - sem')
        plt.plot(dat_le_agg, label = 'long episode mean')
        plt.plot(dat_le_agg + dat_le_agg_ste,linestyle='dashed', label = 'long episode mean + sem')
        plt.plot(dat_le_agg - dat_le_agg_ste,linestyle='dashed', label = 'long episode mean - sem')
        #plt.title('Patient {0}: period {1} - {2}'.format(ptid, period_start, period_end))
        plt.title('Patient {0}'.format(ptid))
        plt.xlabel('epoch')
        plt.ylabel('count(nomalized)')
        plt.tight_layout()
        plt.legend()
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


def plot_roc_all(X_test, y_test, pat):
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
        plt.title('Receiver operating characteristic curve', fontsize=hp.label_fontsize)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.0 , box.width, box.height * 1])
    # ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
    #           shadow = True, ncol = 4, prop = {'size':10})
    plt.legend(prop={'size': 14})
    plt.savefig(hp.prepath_cp + pat.id + '_' + 'roc')
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
    print(labels)
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

def feature_importance(pat, classifier_int, label):
    
    int2name = hp.int2name
    clf_name = int2name[classifier_int]
    clf = pat.estimator[classifier_int]
    classifier_type1 = [1]
    classifier_type2 = [6,7]
    if classifier_int in classifier_type1:
        coef = clf.coef_.reshape(7,4)
    elif classifier_int in classifier_type2:
        coef = np.abs(clf.feature_importances_.reshape(7,4))
    powerband = ['delta', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma', 'all'][::-1]
    channel = ['1', '2', '3', '4']
    df = pd.DataFrame(coef, index = powerband, columns = channel)
    import seaborn as sns
    fig = plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    r = sns.heatmap(coef, cmap=plt.cm.Blues)
    r.set_title("Feature importance heatmap of {} for {}".format(clf_name, label), fontsize=hp.label_fontsize -1)
    ax.set_yticklabels(df.index, fontsize=hp.label_fontsize-5)
    ax.set_xticklabels(df.columns, fontsize=hp.label_fontsize-2)

    plt.savefig(hp.prepath_cp + label + '_' + 'fi')
    plt.show()


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, label = None,**kwargs):
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
    plt.savefig(hp.prepath_sleep + label)
    plt.show()
    return 

