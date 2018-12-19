#/usr/bin/python2


from datetime import datetime
class Hyperparams:
    '''Hyperparameters'''
    col_rs = 'region_start_time'
    col_es = 'episode_starts'
    col_le = 'long_episodes'


    prepath_pat = '../patients/'
    prepath_sleep = '../fig/sleep/'
    num_classifier = 7

    param_outliers = 5

    if_scaler = 1
    if_remove_icd = 1

    int2name = {1:'Logistic Regression', 2: 'SVM', 3: 'Gaussian Naive Bayes classifier', 4:'Linear Discriminant Analysis', 5:'decision tree', 6:'random forest', 7:'gradient boosting'}


    #produce column name including filename, powerband for four channels and interictal discharges
    powerbands1 = ['theta', 'alpha', 'beta', 'lowgamma', 'highgamma', 'all']
    powerbands = ['delta', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma', 'all']
    channel = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']    
    col_names = ['filename', col_rs, 'long_epi', 'sleep']
    for powerband in powerbands:
        for i in range(1,5):
            col_names.append(powerband+str(i))
    col_names.append('i12')
    col_names.append('i34')   

    # outlier drop list
    drop_list = ['filename','label', 'region_start_time', 'id', 'epoch', 'if_stimulated', 'i12', 'i34',]


    label_fontsize = 22