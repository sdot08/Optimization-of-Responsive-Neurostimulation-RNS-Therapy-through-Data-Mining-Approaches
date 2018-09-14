#/usr/bin/python2


from datetime import datetime
class Hyperparams:
    '''Hyperparameters'''
    col_rs = 'region_start_time'
    col_es = 'episode_starts'
    col_le = 'long_episodes'


    prepath_pat = '../patients/'

    num_classifier = 7

    param_outliers = 6

    #produce column name including filename, powerband for four channels and interictal discharges
    powbands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma', 'all']
    col_names = ['filename', col_rs]
    for powband in powbands:
        for i in range(1,5):
            col_names.append(powband+str(i))
    col_names.append('i12')
    col_names.append('i34')   


