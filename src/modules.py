import pandas as pd
import numpy as np
import h5py
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


import jj_basic_fn as JJ
from hyperparams import Hyperparams as hp
from patient import patient
import prep
import plot_funcs 
import sys

def build_patients(index = -1, freq_idx = 0):
    col_rs = hp.col_rs
    col_es = hp.col_es
    col_le = hp.col_le


    from datetime import datetime
    from datetime import time
    #build patient object
    p222_1 = patient('222_1')
    p222_2 = patient('222_2')
    p222_3 = patient('222_3')
    p231 = patient('231')
    # local means whether to use local(weekly) median as threshold
    p229 = patient('229')
    p241 = patient('241')
    p226 = patient('226')

    #add epoch info
    start_222_1 = datetime.strptime('Feb 12 2016', '%b %d %Y')
    end_222_1 = datetime.strptime('Oct 24 2016', '%b %d %Y')
    num_per_epoch_222_1 = 31

    start_222_2 = datetime.strptime('Oct 26 2016', '%b %d %Y')
    end_222_2 = datetime.strptime('May 29 2017', '%b %d %Y')
    num_per_epoch_222_2 = 30
    

    start_222_3 = datetime.strptime('Sep 19 2017', '%b %d %Y')
    end_222_3 = datetime.strptime('Jan 30 2018', '%b %d %Y')
    num_per_epoch_222_3 = 31

    start_222 = [start_222_1, start_222_2, start_222_3]
    end_222 = [end_222_1, end_222_2, end_222_3]
    num_per_epoch_222 = [num_per_epoch_222_1, num_per_epoch_222_2, num_per_epoch_222_3]

    start_231 = datetime.strptime('Feb 7 2017', '%b %d %Y')
    end_231 = datetime.strptime('Feb 21 2018', '%b %d %Y')
    num_per_epoch_231 = 31

    start_229 = datetime.strptime('Oct 9 2017', '%b %d %Y')
    end_229 = datetime.strptime('Aug 23 2018', '%b %d %Y')
    num_per_epoch_229 = 31

    start_241 = datetime.strptime('Nov 14 2017', '%b %d %Y')
    end_241 = datetime.strptime('Oct 4 2018', '%b %d %Y')
    num_per_epoch_241 = 29

    start_226 = datetime.strptime('Apr 26 2017', '%b %d %Y')
    end_226 = datetime.strptime('Nov 12 2017', '%b %d %Y')
    #end_226 = datetime.strptime('Oct 28 2018', '%b %d %Y')

    num_per_epoch_226 = 33

    p231.add_epochinfo(start_231, end_231, num_per_epoch_231)
    p222_1.add_epochinfo(start_222_1, end_222_1, num_per_epoch_222_1)
    p222_2.add_epochinfo(start_222_2, end_222_2, num_per_epoch_222_2)
    p222_3.add_epochinfo(start_222_3, end_222_3, num_per_epoch_222_3)
    p229.add_epochinfo(start_229, end_229, num_per_epoch_229)
    p241.add_epochinfo(start_241, end_241, num_per_epoch_241)
    p226.add_epochinfo(start_226, end_226, num_per_epoch_226)

    #add duration and daily
    prepath = '../data/'

    duration_222_1 = pd.read_csv(prepath + 'NY222_3760120_2015-08-11_to_2016-08-11_20180809165417.csv', skiprows=3)
    duration_222_2 = pd.read_csv(prepath + 'NY222_3760120_2016-08-11_to_2017-08-11_20180809003645.csv', skiprows=3)
    duration_222_3 = pd.read_csv(prepath + 'NY222_3760120_2017-08-11_to_2018-08-08_20180809004150.csv', skiprows=3)
    duration_231_1 = pd.read_csv(prepath + 'NY231_4086548_2016-07-05_to_2017-07-05_20180809005536.csv', skiprows=3)
    duration_231_2 = pd.read_csv(prepath + 'NY231_4086548_2017-07-05_to_2018-08-08_20180809010451.csv', skiprows=3)
    duration_229_1 = pd.read_csv(prepath + '229_duration.csv', skiprows=3)


    daily={}
    daily['222'] = prep.prep_daily(pd.read_csv(prepath + 'NY222_2015-08-11_to_2018-06-12_daily_20180613153105.csv', skiprows=3))
    daily['231'] = prep.prep_daily(pd.read_csv(prepath + 'NY231_2016-07-05_to_2018-06-12_daily_20180613153815.csv', skiprows=3))
    daily['229'] = prep.prep_daily(pd.read_csv(prepath + 'NY229_2017-05-12_to_2018-09-07_daily_20180907183334.csv', skiprows=3))
    daily['241'] = prep.prep_daily(pd.read_csv(prepath + 'NY241_2017-06-13_to_2018-10-05_daily_20181005204526.csv', skiprows=3))
    daily['226'] = prep.prep_daily(pd.read_csv(prepath + 'NY226_2016-02-09_to_2018-10-05_daily_20181005204146.csv', skiprows=3))

    
    #all patients
    if index == -1:
        pat_list = [p231, p222_1, p222_2, p229]
    elif index == 231:
        pat_list = [p231]
    elif index == 2221:
        pat_list = [p222_1]
    elif index == 241:
        pat_list = [p241]
    elif index == 2223:
        pat_list = [p222_3]
    elif index == -2:
        pat_list = [p241, p226]
    for pat in pat_list:  
        if freq_idx == 124:  
            f = h5py.File('../data/features_124' + pat.pat_id + '.mat', 'r')
        elif freq_idx == 90:
            f = h5py.File('../data/features_90' + pat.pat_id + '.mat', 'r')
        else:
            sys.exit(1)
        pat.add_daily(daily[pat.pat_id])
        pat.add_features(f)
        pat.ngood = pat.features.loc[pat.features['label'] == True].shape[0]
        pat.nbad = pat.features.loc[pat.features['label'] == False].shape[0]
        pat.ndata = pat.features.shape[0]
    if len(pat_list) == 1:
        return pat_list[0]
    return tuple(pat_list)


def remove_outliers(dat, thres = 5000):
    num_dat = dat.shape[0]


    output = dat.copy()
    # outlier drop list
    drop_list = ['filename','label', 'region_start_time', 'id', 'epoch', 'if_stimulated', 'i12', 'i34',]
    if 'sleep' in dat.columns:
        drop_list.append('sleep')
    if 'long_epi' in dat.columns:
        drop_list.append('long_epi')
        
    for col in dat.drop(drop_list, axis = 1).columns.values:
        bol = dat.loc[:, col] - np.mean(dat.loc[:, col]) < hp.param_outliers * dat.loc[:, col].std()
        output = output.loc[bol,:]
    num_output = output.shape[0]
    print('Total outliers removed: {}'.format(num_dat - num_output))

    return output

def get_ml_data(pat, test_size = 0.2, if_stimulated = 'all', if_scaler = 1, if_remove_icd = 1, if_remove_sleep = 1, if_remove_le = 1, random_state=42, sleep_class = None, le_class = None, if_remove_delta = 1, if_remove_outliers = 0):
    dat_0 = pat.features
    if sleep_class == 0:
        dat_0 = dat_0.loc[dat_0.loc[:,'sleep'] == 0,:]
    elif sleep_class == 1:
        dat_0 = dat_0.loc[dat_0.loc[:,'sleep'] == 1,:]
    if le_class == 0:
        dat_0 = dat_0.loc[dat_0.loc[:,'long_epi'] == 0,:]
    elif le_class == 1:
        dat_0 = dat_0.loc[dat_0.loc[:,'long_epi'] == 1,:]
    # remove outliers
    if if_remove_outliers:
        dat = remove_outliers(dat_0)
    else:
        dat = dat_0
    y = dat.loc[:,'label']
    drop_list = ['label', 'region_start_time', 'epoch', 'if_stimulated', 'filename', 'id',]
    if if_remove_delta:
        drop_list += ['delta1',  'delta2',  'delta3', 'delta4']
    if if_remove_icd:
        drop_list.append('i12')
        drop_list.append('i34')
    if if_remove_sleep:
        if 'sleep' in dat.columns:
            drop_list.append('sleep')
    if if_remove_le:
        if 'long_epi' in dat.columns:
            drop_list.append('long_epi')
    X = dat.drop(drop_list, axis = 1, inplace = False)
    y=y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state =random_state)
    scaler = preprocessing.StandardScaler().fit(X_train)
    if if_scaler:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)    
    
        
    return X_train, X_test, y_train, y_test


