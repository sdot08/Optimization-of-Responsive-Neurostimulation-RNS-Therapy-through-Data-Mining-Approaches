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


def build_patients():
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

    pat_list = [p231, p222_1, p222_2, p222_3]
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

    p231.add_epochinfo(start_231, end_231, num_per_epoch_231)
    p222_1.add_epochinfo(start_222_1, end_222_1, num_per_epoch_222_1)
    p222_2.add_epochinfo(start_222_2, end_222_2, num_per_epoch_222_2)
    p222_3.add_epochinfo(start_222_3, end_222_3, num_per_epoch_222_3)

    #add duration and daily
    prepath = '../data/'

    duration_222_1 = pd.read_csv(prepath + 'NY222_3760120_2015-08-11_to_2016-08-11_20180809165417.csv', skiprows=3)
    duration_222_2 = pd.read_csv(prepath + 'NY222_3760120_2016-08-11_to_2017-08-11_20180809003645.csv', skiprows=3)
    duration_222_3 = pd.read_csv(prepath + 'NY222_3760120_2017-08-11_to_2018-08-08_20180809004150.csv', skiprows=3)
    duration_231_1 = pd.read_csv(prepath + 'NY231_4086548_2016-07-05_to_2017-07-05_20180809005536.csv', skiprows=3)
    duration_231_2 = pd.read_csv(prepath + 'NY231_4086548_2017-07-05_to_2018-08-08_20180809010451.csv', skiprows=3)

    duration_222 = prep.prep_duration(pd.concat([duration_222_1, duration_222_2, duration_222_3]))
    duration_231 = prep.prep_duration(pd.concat([duration_231_1, duration_231_2]))

    daily_222 = prep.prep_daily(pd.read_csv(prepath + 'NY222_2015-08-11_to_2018-06-12_daily_20180613153105.csv', skiprows=3))
    daily_231 = prep.prep_daily(pd.read_csv(prepath + 'NY231_2016-07-05_to_2018-06-12_daily_20180613153815.csv', skiprows=3))
    
    #str2date, add date column, logduration
    p222_1.add_duration(prep.prep_duration(duration_222))
    p222_2.add_duration(prep.prep_duration(duration_222))
    p222_3.add_duration(prep.prep_duration(duration_222))
    p231.add_duration(duration_231)
    #str2date, add date column
    p222_1.add_daily(daily_222)
    p222_2.add_daily(daily_222)
    p222_3.add_daily(daily_222)
    p231.add_daily(daily_231)

    f = h5py.File('../data/features.mat', 'r')
    f_s = h5py.File('../data/features_sti.mat', 'r')
    
    for pat in pat_list:	
    	pat.add_features(f, f_s)
    return p231, p222_1, p222_2, p222_3


def remove_outliers(dat, thres = 5000):
    num_dat = dat.shape[0]
    drop_list = ['filename','label', 'region_start_time', 'id', 'epoch', 'if_stimulated', 'i12', 'i34']
    for col in dat.drop(drop_list, axis = 1).columns.values:
        bol = dat.loc[:, col] - np.mean(dat.loc[:, col]) < hp.param_outliers * dat.loc[:, col].std()
        dat = dat.loc[bol,:]
    bol = dat.loc[:, 'beta2'] < 400
    dat = dat.loc[bol,:]
    num_output = dat.shape[0]
    print('Total outliers removed: {}'.format(num_dat - num_output))

    return dat

def get_ml_data(pat, test_size = 0.2, if_stimulated = 'all', if_scaler = 1, if_remove_icd = 1, random_state=42):
    dat_0 = pat.features
    dat = remove_outliers(dat_0)
    y = dat.loc[:,'label']
    drop_list = ['label', 'region_start_time', 'epoch', 'if_stimulated', 'filename', 'id']
    if if_remove_icd:
        drop_list.append('i12')
        drop_list.append('i34')
    X = dat.drop(drop_list, axis = 1, inplace = False)
    
    y=y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state =random_state)
    scaler = preprocessing.StandardScaler().fit(X_train)
    if if_scaler:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)    
    
        
    return X_train, X_test, y_train, y_test


