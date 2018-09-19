#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.chdir('/Users/hp/GitHub/EEG/src')
import numpy as np
import math
import pandas as pd
import logging
import sys
import matplotlib.pyplot as plt
import time
from random import shuffle
import itertools
import pickle
from numpy.linalg import inv
import operator
import seaborn as sns
from datetime import datetime, date, time

import jj_basic_fn as JJ
from hyperparams import Hyperparams as hp
from patient import patient
import prep
import pickle
from modules import *
import warnings
warnings.filterwarnings("ignore")
#default size of the graph
plt.rcParams['figure.figsize'] = (10.0, 8.0) 

col_rs = hp.col_rs
col_es = hp.col_es

prepath = '../data/'
p231, p222_1, p222_2, p222_3 = build_patients()
epoch_label_dict = pickle.load(open("epoch_label_dict.p", "rb" ))

logduration_date = p231.duration.loc[:,['duration','date']]
logduration_date.loc[:,'date'] = np.array([datetime.combine(item, datetime.min.time()) for item in logduration_date.date])

logduration_date_epoch = prep.addepoch(logduration_date, 'date', p231.epoch_info['start'], p231.epoch_info['end'], p231.epoch_info['num_per_epoch'])
#put labels according to the epoch
logduration_date_epoch.loc[:,'label'] = logduration_date_epoch.loc[:,'epoch'].apply(lambda x: epoch_label_dict['231'][x])
plt.figure(figsize = (8, 8))
epd = epoch_label_dict['231']


for epoch in range(len(epd)):
    subset = logduration_date_epoch[logduration_date_epoch['epoch'] == epoch]
    color = 'r' if epd[epoch] == True else 'b'
    # Draw the density plot
    sns.distplot(subset['duration'], hist = True, bins = 10, color = color)
plt.xlim(50,400)
plt.ylim(0,0.15)
plt.show()
        