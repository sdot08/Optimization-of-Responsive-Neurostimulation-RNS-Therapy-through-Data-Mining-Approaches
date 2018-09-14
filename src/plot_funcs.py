import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from matplotlib import cm as cm
import numpy as np
import math
import pandas as pd

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


def plot_roc_all(X_test, y_test, patid):
    classifier_list = [1,2,5,6,7]
    cmap = get_cmap(len(classifier_list))
    lw = 2
    plt.figure()
    ax = plt.subplot(111)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    for i,classifier_int in enumerate(classifier_list):

        y_score, accuracy, y_pred, clf_name = load_score(classifier_int, X_test, y_test, patid)
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
