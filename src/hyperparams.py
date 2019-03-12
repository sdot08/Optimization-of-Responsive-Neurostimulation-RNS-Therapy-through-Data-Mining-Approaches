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

    int2name = {1:'Logistic Regression', 2: 'SVM', 3: 'Gaussian Naive Bayes classifier', 4:'Linear Discriminant Analysis', 5:'Decision Tree', 6:'Random Forest', 7:'Gradient Boosting'}


    #produce column name including filename, powerband for four channels and interictal discharges
    powerbands1 = ['Theta', 'Alpha', 'Beta', 'LowGamma', 'HighGamma', 'Broadband']
    powerbands = ['Delta', 'Theta', 'Alpha', 'Beta', 'LowGamma', 'HighGamma', 'Broadband']
    channel = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
    channel_plv = ['Ch1-Ch2', 'Ch1-Ch3', 'Ch1-Ch4', 'Ch2-Ch3', 'Ch2-Ch4', 'Ch3-Ch4']
    #if you change this, remember to change the feature_name = hp.col_names[ind + 4] too
    col_names = ['filename', col_rs, 'long_epi', 'sleep']
    for powerband in powerbands:
        for i in range(1,5):
            col_names.append(powerband+str(i))
  

    col_names_P = col_names.copy()
    channel_pairs = ['Ch1-Ch2', 'Ch1-Ch3', 'Ch1-Ch4', 'Ch2-Ch3', 'Ch2-Ch4', 'Ch3-Ch4']
    for channel_pair in channel_pairs:
        for powerband in powerbands[:-1]:
            cname_P = powerband + '_' + channel_pair 
            col_names_P.append(cname_P)
    
    col_names.append('i12')
    col_names.append('i34') 

    col_names_P.append('i12')
    col_names_P.append('i34') 


    drop_list_all = ['label', 'region_start_time', 'epoch', 'if_stimulated', 'filename', 'id','delta1',  'delta2',  'delta3', 'delta4', 'i12', 'i34', 'sleep', 'long_epi']



    label_fontsize = 22