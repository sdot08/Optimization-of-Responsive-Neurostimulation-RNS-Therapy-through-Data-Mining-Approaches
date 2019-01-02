from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta

import jj_basic_fn as JJ
import pandas as pd
import numpy as np
from hyperparams import Hyperparams as hp


col_rs = hp.col_rs
col_es = hp.col_es
col_le = hp.col_le

#str2date, add date column
def prep_daily(dat):
    # convert string to date
    dat = JJ.df_str2date(dat, hp.col_rs)
    # add the date of the datatime to the new date column
    dat.loc[:,'date'] = dat.loc[:,hp.col_rs].apply(lambda x: x.date())
    return dat

#str2date, add date column, logduration
def prep_duration(dat):
    # convert string to date
    dat = JJ.df_str2date(dat, hp.col_rs)
    # add the date of the datatime to the new date column
    dat.loc[:,'date'] = dat.loc[:,hp.col_rs].apply(lambda x: x.date())
    # add the log of the duration to the new logduration column
    dat.loc[:,'logduration'] = dat.loc[:,'duration'].apply(lambda x: np.log(x))
    return dat

#usage dat.loc[filtertime(dat, col, dstart, dend),:]
def filtertime(dat, col, start_date, end_date):
    return np.array(start_date <= dat[col]) & \
         np.array(dat[col] <= end_date)

def addepoch(dat, col, start, end, num_per_epoch):
    c = end - start
    num_epoch  = int(np.floor(c.days/num_per_epoch))
    start_dates = [start + timedelta(days = i * num_per_epoch) for i in range(num_epoch)]
    end_dates = [start + timedelta(days = (i + 1) * num_per_epoch) for i in range(num_epoch)]
    dates = list(zip(start_dates, end_dates))
    end = dates[-1][-1]
    for i in range(num_epoch):
        date_tup = dates[i]
        dstart, dend = date_tup[0],date_tup[1]
        dat.loc[filtertime(dat, col, dstart, dend),'epoch'] = int(i)

    data = dat[np.array(start <= dat.loc[:,col]) & \
         np.array(dat.loc[:,col] <= end)]
    data.loc[:,'epoch'] = data.loc[:,'epoch'].astype(int)
    return data

def firstnorm(dat, col):
    base = np.array(dat[col])[0]
    if base == 0:
        base = 1

    dat.loc[:,col] = dat[col] / base
    return dat, base

def prep_daily2(dat, pat):
    #output processes dataframe and epoch_label dict with key = patid
    data_2_dict, epoch_label_dict, epoch_label_epi_dict = {}, {}, {}
    dat.loc[:,col_rs] = pd.to_datetime(dat.loc[:,col_rs])
    for pat in pat_list:
        pat_id = pat.id
        epoch_info = pat.epoch_info
        data_0 = dat.loc[dat.loc[:,'id'] == pat_id,:]
        data_1 = addepoch(dat, col_rs, epoch_info['start'], epoch_info['end'], epoch_info['num_per_epoch'])
        data_2, epoch_label_var, epoch_label_epi_var = epoch_label(data_1)
        data_2_dict[pat_id] = data_2
        epoch_label_epi_dict[pat_id] = epoch_label_epi_var
        epoch_label_dict[pat_id] = epoch_label_var
    return data_2_dict, epoch_label_dict, epoch_label_epi_dict

def epoch_label(dat):
    #output dataframe with columns labels and labels_epi, dictionary(key: epoch, val: label)
    pd.set_option('display.max_rows', 1000)

    #print(pd.DataFrame(dat.loc[:, [col_le, 'epoch']]).groupby('epoch').agg('mean'))
    dat_epi_agg, dat_le_agg, dat_epi_agg_ste, dat_le_agg_ste = dat_agg(dat)
    n = dat_epi_agg.shape[0]

    # generate label according to episode counts
    thres_epi = np.median(dat_epi_agg)

    keys_epi = list(np.array(dat_epi_agg.index, dtype = int))
    vals_epi = list(np.array(dat_epi_agg.loc[:,col_es] < thres_epi))
    epoch_label_epi = dict(zip(keys_epi, vals_epi))
    for key_epi in epoch_label_epi:
        val_epi = epoch_label_epi[key_epi]
        dat.loc[dat['epoch'] == key_epi,'label_epi'] = val_epi
    # generate label according to long episode
    thres = np.median(dat_le_agg)
    keys = list(np.array(dat_le_agg.index, dtype = int))
    vals = list(np.array(dat_le_agg.loc[:,col_le] < thres))
    epoch_label = dict(zip(keys, vals))
    for key in epoch_label:
        val = epoch_label[key]
        dat.loc[dat['epoch'] == key,'label'] = val
    return dat, epoch_label, epoch_label_epi




def dat_agg(dat):
    N = dat.shape[0]
    dat_epi_agg = dat.loc[:, [col_es, 'epoch']].groupby('epoch').agg('mean')
    dat_epi_agg, base_epi = firstnorm(dat_epi_agg, col_es)
    dat_le_agg = dat.loc[:, [col_le, 'epoch']].groupby('epoch').agg('mean')
    # dat_le_agg, base_le = firstnorm(dat_le_agg, col_le)
    dat_epi_agg_ste = dat.loc[:, [col_es, 'epoch']].groupby('epoch').std()/base_epi/np.sqrt(N)
    #dat_le_agg_ste = dat.loc[:, [col_le, 'epoch']].groupby('epoch').std()/base_le/np.sqrt(N)
    dat_le_agg_ste = dat.loc[:, [col_le, 'epoch']].groupby('epoch').std()/np.sqrt(N)
    
    return dat_epi_agg, dat_le_agg, dat_epi_agg_ste, dat_le_agg_ste

