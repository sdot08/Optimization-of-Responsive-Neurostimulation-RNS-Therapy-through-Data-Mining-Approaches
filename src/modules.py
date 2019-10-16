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
from datetime import datetime
from datetime import time

np.random.seed(42)

def build_patients(index = -1, freq_idx = 0, if_weekly = 0, if_2weekly = 0, if_PSV = 0, if_yrandom = 0, if_sliding_window = 0, sw_size = 31, log = 0, if_shuffle_label = 0, reg = 0):
    col_rs = hp.col_rs
    col_es = hp.col_es
    col_le = hp.col_le



    #build patient object
    p222_1 = patient('222_1', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p222_2 = patient('222_2', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p222_3 = patient('222_3', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p222 = patient('222', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p231 = patient('231', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # local means whether to use local(weekly) median as threshold
    # p229 = patient('229', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p241 = patient('241', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)

    # 06/2019   239,225,251,217,229
    # p239_1 = patient('239_1', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # p239_2 = patient('239_2', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # p225_1 = patient('225_1', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # p225_2 = patient('225_2', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # p251 = patient('251', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # p217 = patient('217', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    # p229 = patient('229', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p239 = patient('239', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p225 = patient('225', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p201 = patient('201', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)
    p226 = patient('226', if_sliding_window = if_sliding_window, sw_size = sw_size, reg = reg)

    #add epoch info
    # start_222_1 = datetime.strptime('Feb 12 2016', '%b %d %Y')
    # end_222_1 = datetime.strptime('Oct 24 2016', '%b %d %Y')
    # num_per_epoch_222_1 = 31

    start_222 = datetime.strptime('Feb 12 2016', '%b %d %Y')
    end_222 = datetime.strptime('May 29 2017', '%b %d %Y')
    num_per_epoch_222 = 31


    start_222_2 = datetime.strptime('Oct 26 2016', '%b %d %Y')
    end_222_2 = datetime.strptime('May 29 2017', '%b %d %Y')
    num_per_epoch_222_2 = 30
    

    start_222_3 = datetime.strptime('Sep 19 2017', '%b %d %Y')
    end_222_3 = datetime.strptime('Jan 30 2018', '%b %d %Y')
    num_per_epoch_222_3 = 31

    # start_222 = [start_222_1, start_222_2, start_222_3]
    # end_222 = [end_222_1, end_222_2, end_222_3]
    # num_per_epoch_222 = [num_per_epoch_222_1, num_per_epoch_222_2, num_per_epoch_222_3]



    start_231 = datetime.strptime('Dec 6 2016', '%b %d %Y')
    end_231 = datetime.strptime('Feb 21 2018', '%b %d %Y')
    num_per_epoch_231 = 31


    # start_231 = datetime.strptime('Feb 7 2017', '%b %d %Y')
    # end_231 = datetime.strptime('Feb 21 2018', '%b %d %Y')
    # num_per_epoch_231 = 31

    start_241 = datetime.strptime('Nov 14 2017', '%b %d %Y')
    end_241 = datetime.strptime('Oct 4 2018', '%b %d %Y')
    num_per_epoch_241 = 29

    start_239 = datetime.strptime('Jan 4 2018', '%b %d %Y')
    end_239 = datetime.strptime('May 24 2019', '%b %d %Y')
    num_per_epoch_239 = 31

    start_225 = datetime.strptime('Jul 19 2016', '%b %d %Y')
    end_225 = datetime.strptime('Jul 12 2017', '%b %d %Y')
    num_per_epoch_225 = 33 

#5/22/2015 - 1/29/2016, 4/11/2017 - 10/22/2017

    start_201 = datetime.strptime('Feb 7 2016', '%b %d %Y')
    end_201 = datetime.strptime('Oct 22 2017', '%b %d %Y')
    num_per_epoch_201 = 31 
    # Apr 26, 2017 to Nov 12, 2017
    start_226 = datetime.strptime('Apr 26 2017', '%b %d %Y')
    end_226 = datetime.strptime('Apr 16 2018', '%b %d %Y')
    num_per_epoch_226 = 32 
    # start_239_1 = datetime.strptime('Jul 13 2018', '%b %d %Y')
    # end_239_1 = datetime.strptime('Jan 31 2019', '%b %d %Y')
    # num_per_epoch_239_1 = 28

    # start_239_2 = datetime.strptime('Jan 4 2018', '%b %d %Y')
    # end_239_2 = datetime.strptime('Jul 11 2018', '%b %d %Y')
    # num_per_epoch_239_2 = 31

    # start_225_1 = datetime.strptime('Aug 24 2016', '%b %d %Y')
    # end_225_1 = datetime.strptime('Apr 12 2017', '%b %d %Y')
    # num_per_epoch_225_1 = 33

    # start_225_2 = datetime.strptime('Oct 13 2017', '%b %d %Y')
    # end_225_2 = datetime.strptime('Jun 5 2018', '%b %d %Y')
    # num_per_epoch_225_2 = 29    

    # start_251 = datetime.strptime('Sep 26 2018', '%b %d %Y')
    # end_251 = datetime.strptime('Apr 28 2019', '%b %d %Y')
    # num_per_epoch_251 = 30

    # start_217 = datetime.strptime('Jul 19 2018', '%b %d %Y')
    # end_217 = datetime.strptime('May 18 2019', '%b %d %Y')
    # num_per_epoch_217 = 30

    # start_229 = datetime.strptime('Oct 9 2017', '%b %d %Y')
    # end_229 = datetime.strptime('Feb 13 2019', '%b %d %Y')
    # num_per_epoch_229 = 27

    p231.add_epochinfo(start_231, end_231, num_per_epoch_231)
    # p222_1.add_epochinfo(start_222_1, end_222_1, num_per_epoch_222_1)
    # p222_2.add_epochinfo(start_222_2, end_222_2, num_per_epoch_222_2)
    # p222_3.add_epochinfo(start_222_3, end_222_3, num_per_epoch_222_3)
    p222.add_epochinfo(start_222, end_222, num_per_epoch_222)
    # p229.add_epochinfo(start_229, end_229, num_per_epoch_229)
    p241.add_epochinfo(start_241, end_241, num_per_epoch_241)


    # p239_1.add_epochinfo(start_239_1, end_239_1, num_per_epoch_239_1)
    # p239_2.add_epochinfo(start_239_2, end_239_2, num_per_epoch_239_2)
    # p225_1.add_epochinfo(start_225_1, end_225_1, num_per_epoch_225_1)
    # p225_2.add_epochinfo(start_225_2, end_225_2, num_per_epoch_225_2)
    # p251.add_epochinfo(start_251, end_251, num_per_epoch_251)
    # p217.add_epochinfo(start_217, end_217, num_per_epoch_217)
    # p229.add_epochinfo(start_229, end_229, num_per_epoch_229)
    p239.add_epochinfo(start_239, end_239, num_per_epoch_239)
    p225.add_epochinfo(start_225, end_225, num_per_epoch_225)
    p201.add_epochinfo(start_201, end_201, num_per_epoch_201)    
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
    # daily['229'] = prep.prep_daily(pd.read_csv(prepath + 'NY229_2017-05-12_to_2018-09-07_daily_20180907183334.csv', skiprows=3))
    daily['241'] = prep.prep_daily(pd.read_csv(prepath + 'NY241_2017-06-13_to_2018-10-05_daily_20181005204526.csv', skiprows=3))
    
    daily['239'] = prep.prep_daily(pd.read_csv(prepath + 'NY239_2017-03-28_to_2019-05-24_daily_20190524174640.csv', skiprows=3))
    daily['225'] = prep.prep_daily(pd.read_csv(prepath + 'NY225_2016-01-14_to_2019-05-23_daily_20190524174650.csv', skiprows=3))
    daily['201'] = prep.prep_daily(pd.read_csv(prepath + '201_daily.csv', skiprows=3))
    daily['226'] = prep.prep_daily(pd.read_csv(prepath + 'NY226_2016-02-09_to_2018-10-05_daily_20181005204146.csv', skiprows=3))


    # daily['251'] = prep.prep_daily(pd.read_csv(prepath + 'NY251_2018-02-06_to_2019-04-29_daily_20190524174653.csv', skiprows=3))
    # daily['217'] = prep.prep_daily(pd.read_csv(prepath + 'NY217_2015-05-14_to_2019-05-18_daily_20190524174658.csv', skiprows=3))
    # daily['229'] = prep.prep_daily(pd.read_csv(prepath + 'NY229_2016-05-12_to_2019-05-20_daily_20190524174704.csv', skiprows=3))

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
        pat_list = [p239,p225,p222]
    elif index == -3:
        pat_list = [p231, p222_1, p222_2, p241]
    elif index == -4:
        pat_list = [p231, p222, p201, p226]


    for pat in pat_list:  
        if if_weekly:
            pat.epoch_info['num_per_epoch'] = 7
        if if_2weekly:
            pat.epoch_info['num_per_epoch'] = 14
        if if_PSV:
            f = h5py.File('../data/features_sync_90' + pat.pat_id + '.mat', 'r')
        elif freq_idx == 124:  
            f = h5py.File('../data/features_124' + pat.pat_id + '.mat', 'r')
        elif freq_idx == 90:
            f = h5py.File('../data/features_90' + pat.pat_id + '.mat', 'r')
        else:
            sys.exit(1)
        
        # use daily file to identify the label for each epoch or each ECog
        pat.add_daily(daily[pat.pat_id], log = log, reg = reg)
        

        # if use random label
        if if_yrandom:
            prev_dict = pat.epoch_label_dict
            pat.prev_dict = prev_dict
            n = len(prev_dict)
            print('prev_dict: ', prev_dict)
            new_dict = {}
            tf = np.array([True, False])
            if pat.id == '241':
                np.random.seed(40)
            new_vals = np.random.choice(tf,size = n,)
            np.random.seed(42)
            for i,key in enumerate(prev_dict):
                new_dict[key] = new_vals[i]
            print('new_dict', new_dict)
            pat.epoch_label_dict = new_dict

        pat.add_features(f, if_PSV = if_PSV)
        if if_shuffle_label:
            label_list = list(pat.features.label)
            np.random.shuffle(label_list)
            pat.features.label = label_list
        pat.ngood = pat.features.loc[pat.features['label'] == True].shape[0]
        pat.nbad = pat.features.loc[pat.features['label'] == False].shape[0]
        pat.ndata = pat.features.shape[0]
        pat.print_features_property()
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

def get_ml_data(pat, test_size = 0.2, if_stimulated = 'all', if_scaler = 1, \
    if_remove_icd = 1, if_remove_sleep = 1, if_remove_le = 1, random_state=42,\
     sleep_class = None, le_class = None, if_remove_delta = 1, if_remove_outliers = 0,\
      if_split = 0, epoch = None, if_reg = 0, test = 0):
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
        dat = dat_0.copy()

    drop_list = ['label', 'region_start_time', 'epoch', 'if_stimulated', 'filename', 'id',]
    if if_remove_delta:
        for x in hp.col_names_P:
            if (x[:5] == 'Delta' or x[:5] == 'delta') and (x in dat_0.columns):
                drop_list.append(x)

    if if_remove_icd:
        drop_list.append('i12')
        drop_list.append('i34')
    if if_remove_sleep:
        if 'sleep' in dat.columns:
            drop_list.append('sleep')
    if if_remove_le:
        if 'long_epi' in dat.columns:
            drop_list.append('long_epi')

    if if_split == 1:
        print('epoch split')
        X = dat
        X_test = X.groupby('epoch', group_keys=False).apply(lambda x: x.sample(frac = 0.2, random_state = random_state))
        train_idx = [a for a in list(X.index) if a not in list(X_test.index)]
        X_train = X.loc[train_idx]
        # print(X_train.shape)
        # print(X_test.shape)
        # print(X.shape)
        y_train = X_train.loc[:,'label'].astype('int')
        y_test = X_test.loc[:,'label'].astype('int')
        X_train = np.array(X_train.drop(drop_list, axis = 1, inplace = False))
        X_test = np.array(X_test.drop(drop_list, axis = 1, inplace = False))
    elif if_split == -1:
        y = dat.loc[:,'label']       
        X = dat
        ## change if reg
        y=y.astype('int')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state =random_state)
        idx = X_test.loc[:, 'epoch'] == epoch
        X_test = np.array(X_test.loc[idx].drop(drop_list, axis = 1, inplace = False))
        X_train = np.array(X_train.drop(drop_list, axis = 1, inplace = False))
        y_test = y_test.loc[idx].astype('int')
    elif if_split == -2:
        X = dat
        X_test = X.groupby('epoch', group_keys=False).apply(lambda x: x.sample(frac = test_size , random_state = random_state))
        train_idx = [a for a in list(X.index) if a not in list(X_test.index)]
        X_train = X.loc[train_idx]
        # print(X_train.shape)
        # print(X_test.shape)
        # print(X.shape)
        y_train = X_train.loc[:,'label'].astype('int')
        y_test = X_test.loc[:,'label'].astype('int')   
    else:
        y = dat.loc[:,'label']       
        X = dat.drop(drop_list, axis = 1, inplace = False)
        if not pat.reg:
            y=y.astype('int')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state =random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state =random_state)
        print('not split')
    if if_scaler and if_split >= -1:

        scaler = preprocessing.StandardScaler().fit(X_train) 
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)    

    if test:
        return X
    return X_train, X_test, y_train, y_test


