import pandas as pd

from prep import *
from hyperparams import Hyperparams as hp

class patient():
    def __init__(self, id):
        self.id = id
        self.pat_id = id.split('_')[0]
        self.duration = None
        self.daily = None
        self.hasduration = False
        self.hasdaily = False
        pass

    def add_epochinfo(self, start, end, num_per_epoch):
        self.epoch_info = {}
        self.epoch_info['start'] = start
        self.epoch_info['end'] = end
        self.epoch_info['num_per_epoch'] = num_per_epoch

    def add_duration(self,dat):
        output = dat.loc[filtertime(dat, hp.col_rs, self.epoch_info['start'], self.epoch_info['end']),:]
        self.duration = output


    def add_daily(self,dat):
        data0 = dat.loc[filtertime(dat, hp.col_rs, self.epoch_info['start'], self.epoch_info['end']),:]
        epoch_info = self.epoch_info
        data_1 = addepoch(dat, hp.col_rs, epoch_info['start'], epoch_info['end'], epoch_info['num_per_epoch'])
        data_2, epoch_label_dict, epoch_label_epi_dict = epoch_label(data_1)
        self.epoch_label_dict = epoch_label_dict
        self.epoch_label_epi_dict = epoch_label_epi_dict
        self.daily = data_2

    def add_features(self, f, f_s):
        #column name including filename, powerband for four channels and interictal discharges
        col_names = hp.col_names
        pat_id = self.pat_id

        matlab_features_name, matlab_sti_features_name = 'T_' + pat_id + '_arr_scheduled', 'T_' + pat_id + '_arr_scheduled_sti'
        features_0 = pd.DataFrame(np.array(f[matlab_features_name]).T, columns = col_names)
        features_sti_0 = pd.DataFrame(np.array(f_s[matlab_sti_features_name]).T, columns = col_names)
        
        features_1 = self.add_features_helper(features_0, False)
        features_sti_1 = self.add_features_helper(features_sti_0, True)


        self.features = features_1
        self.features_sti = features_sti_1


    def add_features_helper(self, fea, if_stimulated):
        features = fea.copy()
        features.loc[:,hp.col_rs] = pd.to_datetime(features.loc[:,col_rs], unit = 'd', origin=pd.Timestamp('2000-01-01'))
        features = addepoch(features, hp.col_rs,self.epoch_info['start'], self.epoch_info['end'], self.epoch_info['num_per_epoch'])
        for key in self.epoch_label_dict:
            val = self.epoch_label_dict[key]
            features.loc[features.loc[:,'epoch'] == key,'label'] = val

        features.loc[:,'id'] = self.id
        features.loc[:,'if_stimulated'] = if_stimulated
        return features



