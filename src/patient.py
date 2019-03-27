import pandas as pd
import numpy as np

import prep 
from hyperparams import Hyperparams as hp
np.random.seed(42)

class patient():
    def __init__(self, id):
        self.id = id  #string id for the patient, example: 222_1. 231
        self.pat_id = id.split('_')[0] #string id for the patient, example: 222, 231
        self.duration = None #dataframe that contain duration information for the patient
        self.daily = None  #dataframe that contain daily count information for the patient
        self.features = None #dataframe that contain features
        self.result = {}  #cross validation results
        self.estimator = {} #best estimator
        self.score = {} #best validation scores
        self.params = {} # params for the estimator
        self.y_random = y_random

    def add_epochinfo(self, start, end, num_per_epoch):
        self.epoch_info = {}
        self.epoch_info['start'] = start
        self.epoch_info['end'] = end
        self.epoch_info['num_per_epoch'] = num_per_epoch
        self.epoch_info['num_epochs'] = int((end - start).days / num_per_epoch)

    def add_duration(self,dat):
        output = dat.loc[prep.filtertime(dat, hp.col_rs, self.epoch_info['start'], self.epoch_info['end']),:]
        self.duration = output

    # add epoch to the dataframe, add label to the dataframe, produce epoch2label dict
    def add_daily(self,dat):
        data0 = dat.loc[prep.filtertime(dat, hp.col_rs, self.epoch_info['start'], self.epoch_info['end']),:]
        epoch_info = self.epoch_info
        data_1 = prep.addepoch(dat, hp.col_rs, epoch_info['start'], epoch_info['end'], epoch_info['num_per_epoch'])
        data_2, epoch_label_dict, epoch_label_epi_dict = prep.epoch_label(data_1)
        self.epoch_label_dict = epoch_label_dict
        self.epoch_label_epi_dict = epoch_label_epi_dict
        self.daily = data_2

    def add_features(self, f, if_PSV = False):
        #column name including filename, powerband for four channels and interictal discharges
        pat_id = self.id
        matlab_features_name = 'T_arr_scheduled'
        if if_PSV:
            col_names = hp.col_names_P
        else:
            col_names = hp.col_names
        a = np.array(f[matlab_features_name]).T
        features_0 = pd.DataFrame(np.array(f[matlab_features_name]).T, columns = col_names)
        # all if_sti is False, for the purpose of convenience, since we are not using them
        features_1 = self.add_features_helper(features_0, False)


        self.features = features_1

    
    def add_features_helper(self, fea, if_stimulated):
        features = fea.copy()
        features.loc[:,hp.col_rs] = pd.to_datetime(features.loc[:,hp.col_rs], unit = 'd', origin=pd.Timestamp('2000-01-01'))
        features = prep.addepoch(features, hp.col_rs,self.epoch_info['start'], self.epoch_info['end'], self.epoch_info['num_per_epoch'])
        for key in self.epoch_label_dict:
            val = self.epoch_label_dict[key]
            features.loc[features.loc[:,'epoch'] == key,'label'] = val

        features.loc[:,'id'] = self.id
        features.loc[:,'if_stimulated'] = if_stimulated
        return features

    def print_features_property(self):
        print(self.id)
        print('good: ', self.ngood)
        print('bad: ', self.nbad)
        print('total: ', self.ndata)


