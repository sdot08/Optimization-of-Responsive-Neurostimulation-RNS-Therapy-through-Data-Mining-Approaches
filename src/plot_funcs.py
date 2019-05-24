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
from sklearn.metrics import accuracy_score 



import jj_basic_fn as JJ
import pandas as pd
import numpy as np
from hyperparams import Hyperparams as hp
import prep
import modules

# input pat, output roc_auc,accuracy for the input classifier, if none, use the best one
def pat_performance(pat, classifier_int = None):
    X_train, X_test, y_train, y_test = pat.X_train, pat.X_test, pat.y_train, pat.y_test
    if classifier_int == None:
        classifier_int = pat.best_estimator
    y_score, accuracy, y_pred, clf_name = JJ.load_score(classifier_int, X_test, y_test, pat)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr) 
    return roc_auc, accuracy
# plot the performance for all the classifiers of all the pat in pats
def scatter_performance_all(pats, if_save = 0):
    classifier_list = [1,2,5,6,7]
    cmap = get_cmap(len(classifier_list) + 1)

    plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots(1,1)
    for i,pat in enumerate(pats):

        for j,classifier_int in enumerate(classifier_list):
            xs = []
            ys = []
            np.random.seed(i+j+12)
            epsilon = (np.random.random() - 0.5) *0.2
            xs.append(i + epsilon)
            roc_auc, accuracy = pat_performance(pat, classifier_int)
            ys.append(roc_auc)
    
            plt.scatter(xs,ys, color = cmap(j),label = hp.int2name[classifier_int],s=70)
            if i == 0:
                plt.legend(fontsize=hp.label_fontsize-8)
    plt.plot(range(len(pats)), [0.7] * len(pats), 'g--')
    plt.xlabel('Patient', fontsize=hp.label_fontsize-2)
    plt.ylabel('AUC', fontsize=hp.label_fontsize-2)
    ax.tick_params(labelsize=hp.label_fontsize-2)
    ax.set_xticks(range(len(pats)))
    ax.set_xticklabels([pat.id for pat in pats], fontsize=hp.label_fontsize-2)
    ax.set_xticklabels([pat.id for pat in pats], fontsize=hp.label_fontsize-2)
    if if_save:
        plt.savefig('../fig/scatter_performance_all.png')
    plt.tight_layout()
    plt.show()

# plot mean long episode for each epoch along with the accuracy of the prediction for each epoch
def plot_epoch_mean_acc(pat_list, if_save = 0, label = '', random_states = None, \
                        if_remove_sleep=1, if_remove_le = 1, le_class = None, \
                        sleep_class =None, if_remove_delta = 1, if_scaler = 1,\
                        if_remove_outliers = 0, if_remove_icd = 1,if_split = 0, legend_list = None,if_title = 1):
    #sample label : '_weekly'
    pat = pat_list[0]
    pat.print_features_property()
    ptid = pat.id
    dat = pat.daily
    dat_epi_agg, dat_le_agg, dat_epi_agg_ste, dat_le_agg_ste = prep.dat_agg(dat)
    xlabel = 'Epoch (month)'
    epoch_label_dict = pat.epoch_label_dict
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
    ax.tick_params(axis='y')
    plt.errorbar(np.array(good_idx), np.array(dat_le_agg.iloc[good_idx]),yerr=np.array(dat_le_agg_ste.iloc[good_idx]), fmt='o', mfc='blue',ecolor='black', markersize='12',label = 'Good')
    plt.errorbar(np.array(bad_idx), np.array(dat_le_agg.iloc[bad_idx]),yerr=np.array(dat_le_agg_ste.iloc[bad_idx]), fmt='o', mfc='red',ecolor='black', markersize='12',label = 'Bad')
    plt.plot(dat_le_agg, label = 'long episode mean', color = 'black', linewidth=3.0)
    ax.set_ylabel('Mean Long Episode Count Per Day', fontsize=hp.label_fontsize)
    ax.set_xlabel(xlabel, fontsize=hp.label_fontsize)
    ax2 = ax.twinx()
    ax2.tick_params(axis='y2', labelcolor='blue') 
    cmap = get_cmap(5,name = 'gist_ncar')
    for i,pat in enumerate(pat_list):
        num_epochs = int((pat.epoch_info['end'] - pat.epoch_info['start']).days / pat.epoch_info['num_per_epoch'])
        acc_list = []
        # X_train, X_test, y_train, y_test = pat.X_train, pat.X_test, pat.y_train, pat.y_test
        # y_pred = pat.estimator[pat.best_estimator].predict(X_test)
        # acc = accuracy_score(y_test, y_pred)
        # print('acc = ', acc)
        if random_states != None:
            random_state = random_states[i]
        else:
            random_state = 42
        for epoch in range(num_epochs):
            X_train, X_test, y_train, y_test = modules.get_ml_data(pat, if_scaler = if_scaler, if_remove_icd = if_remove_icd, if_remove_sleep = if_remove_sleep, if_remove_le = if_remove_le, sleep_class = sleep_class, le_class = le_class, if_remove_delta = if_remove_delta, if_remove_outliers = if_remove_outliers, if_split = if_split, random_state =random_state)
            y_pred = pat.estimator[pat.best_estimator].predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)
        label = None
        if legend_list != None:
            label = legend_list[i] + '   AUC = ' + str(JJ.scores_estimators(pat.X_test, pat.y_test, pat, if_show = 0, if_auc = 1))[:5]

        #ax2.scatter(range(num_epochs),acc_list, label = label)
        ax2.plot(range(num_epochs),acc_list, label = 'Run ' + str(i + 1), color = cmap(i))
        #print(np.mean(acc_list))
    ax2.plot(range(num_epochs), [0.5] * num_epochs, 'g--')

    ax2.set_ylabel('Accuracy', fontsize=hp.label_fontsize)
    if if_title:
        plt.title('Best model with different split for Patient {0}'.format(ptid), fontsize=hp.label_fontsize)
    ax2.set_xlabel(xlabel, fontsize=hp.label_fontsize)
    #plt.ylabel('mean long episode count per day', fontsize=hp.label_fontsize)
    plt.ylim(0,1.05)
    plt.legend()
    plt.tight_layout()
    if if_save:
        directory = '../fig/performance variance'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + '/' + pat.id + ' performance variance ' + '.png')
    plt.show()




def plot_epoch_mean(pat_list, if_save = 0, label = '', if_title = 1, if_yrandom = 0):
    #sample label : '_weekly'
    for pat in pat_list:
        pat.print_features_property()
        ptid = pat.id
        dat = pat.daily
        dat_epi_agg, dat_le_agg, dat_epi_agg_ste, dat_le_agg_ste = prep.dat_agg(dat)

        xlabel = 'Epoch (month)'
        good_idx = []
        bad_idx = []

        if not if_yrandom:
            epoch_label_dict = pat.epoch_label_dict
            for key, val in epoch_label_dict.items():
                # colors.append('green' if val else 'red')
                if val:
                    good_idx.append(key)
                else:
                    bad_idx.append(key)
        else:
            epoch_label_dict = pat.prev_dict
            for key, val in epoch_label_dict.items():
                # colors.append('green' if val else 'red')
                if val:
                    good_idx.append(key)
                else:
                    bad_idx.append(key)
            good_r = []
            bad_r = []
            for key, val in pat.epoch_label_dict.items():
                # colors.append('green' if val else 'red')
                if val:
                    good_r.append(key)
                else:
                    bad_r.append(key)   

        colors = [] #colors for plt.bar
        fig, ax = plt.subplots(1,1)
        ax.set_xticks(range(dat_le_agg.shape[0]))
        ax.set_xticklabels(range(1,dat_le_agg.shape[0] + 1))
        
        plt.errorbar(np.array(good_idx), np.array(dat_le_agg.iloc[good_idx]),yerr=np.array(dat_le_agg_ste.iloc[good_idx]), fmt='o', mfc='blue',ecolor='black', markersize='12',label = 'Good')
        plt.errorbar(np.array(bad_idx), np.array(dat_le_agg.iloc[bad_idx]),yerr=np.array(dat_le_agg_ste.iloc[bad_idx]), fmt='o', mfc='red',ecolor='black', markersize='12',label = 'Bad')
        plt.plot(dat_le_agg, label = 'Long Episode Mean', color = 'black')
        if if_yrandom:
            plt.errorbar(np.array(good_r), np.array([1] * len(good_r)),yerr=np.array([0] * len(good_r)), fmt='o', mfc='blue',ecolor='black', markersize='12',label = 'Good')
            plt.errorbar(np.array(bad_r), np.array([1] * len(bad_r)),yerr=np.array([0] * len(bad_r)), fmt='o', mfc='red',ecolor='black', markersize='12',label = 'Bad')
        if if_title:
            plt.title('Epoch Label for Patient {0}'.format(ptid), fontsize=hp.label_fontsize)
        

        plt.xlabel(xlabel, fontsize=hp.label_fontsize)
        plt.ylabel('Mean Long Episode Count Per Day', fontsize=hp.label_fontsize)
        plt.tight_layout()
        plt.legend(fontsize=hp.label_fontsize-2)
        if if_save:
            directory = '../fig/mean_long_episode_count'
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(directory + '/' + pat.id + label + 'mean_long_episode_count' + '.png')
        
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


def plot_roc_all(X_test, y_test, pat, if_save = 0, if_title = 1):
    classifier_list = [1,2,5,6,7]
    cmap = get_cmap(len(classifier_list) + 1)
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
        if if_title:
            plt.title('Patient ' + pat.id + ': ROC for All Classifiers', fontsize=hp.label_fontsize)
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
    return plt.cm.get_cmap(name, n)

def fi_plv(co):
    # 24 + 6 * 6 = 60
    output = np.zeros(60)
    for i in range(6):
        for j in range(4):
            output[i*10 + j] = co[i*4 + j]
    # loop through 5 bandwidths
    for i in range(5):
        # loop through 6 pairs of channels
        for j in range(6):
            output[i * 10 + 4 + j] = co[24 + 5*j + i]
    output[54:] = 0
    return output

def feature_importance(pat, classifier_int, if_save = 0, if_abs = 1, if_title = 1, if_plv = 0, if_sleep = 0):
    int2name = hp.int2name
    clf_name = int2name[classifier_int]
    clf = pat.estimator[classifier_int]
    classifier_type1 = [1]
    classifier_type2 = [6,7]
    topk = 2 #print topk important features
    if if_plv:
        dim1 = 10
    else:
        dim1 = 4

    if classifier_int in classifier_type1:
            print(clf.coef_)
            if if_plv:
                co = fi_plv(clf.coef_[0])
            else:
                co = clf.coef_
            vmin = None
            vmax = None

    elif classifier_int in classifier_type2: 
            if if_plv:
                co = fi_plv(clf.feature_importances_)
            else:
                co = clf.feature_importances_
            vmin = 0
            vmax = 0.08

    coef = co.reshape(6,dim1)
    if if_abs == 1:
        coef = np.abs(coef)
        cmap = plt.cm.Blues
    else:
        cmap = plt.cm.Reds
    if if_plv:
        df = pd.DataFrame(coef, index = hp.powerbands1, columns = hp.channel_plv)
    else:
        df = pd.DataFrame(coef, index = hp.powerbands1, columns = hp.channel)
    
    import seaborn as sns
    
    fig = plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    r = sns.heatmap(coef, cmap=cmap, vmin = vmin, vmax = vmax)
    label = 'Patient '+ pat.id
    if if_title:
        #r.set_title("Feature Importance Heatmap of {} for {}".format(clf_name, label), fontsize=hp.label_fontsize -2)
        r.set_title("{}".format(clf_name), fontsize=hp.label_fontsize -2)
    ax.set_yticklabels(df.index, fontsize=hp.label_fontsize-8, verticalalignment = 'center')
    ax.set_xticklabels(df.columns, fontsize=hp.label_fontsize-15)
    if if_save:
        plt.savefig('../fig/'+ pat.id + '/fi_' + clf_name + '.png')
    plt.show()




    # inds = np.argsort(coef.ravel())[-topk:]
    # feature_names = []
    # for ind in inds:
    #     #feature_name = ', ' + hp.powerbands1[ind // 4] + ' ' + hp.channel[ind % 4] + ' '
    #     feature_name = hp.col_names[ind + 8]
    #     feature_names.append(feature_name)
    # print_statement = 'The 3 most important features for ' + str(clf_name) + ' are ' 
    # if classifier_int == 1:
    #     pat.topfeatures_1 = [feature_names[j] for j in range(topk)]
    #     pat.topfeatures_1 = pat.topfeatures_1[::-1]
    # elif classifier_int == 7:
    #     pat.topfeatures_7 = [feature_names[j] for j in range(topk)]
    #     pat.topfeatures_7 = pat.topfeatures_7[::-1]
    # for i in range(topk):
    #     print_statement += feature_names[i] + ', '
    
    
    # print(pat.topfeatures_1)
    # print(print_statement)
    # print(coef)





def render_mpl_table(data, pat, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, label = None, if_title = 1, if_save = 0, **kwargs):
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
        if if_title:
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



