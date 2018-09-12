from hyperparams import Hyperparams as hp
from patient import patient
import prep
import pandas as pd
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
	p222_1.add_epochinfo(start_222_1, end_222, num_per_epoch_222_1)
	p222_2.add_epochinfo(start_222_2, end_222_2, num_per_epoch_222_2)
	p222_3.add_epochinfo(start_222_3, end_222_3, num_per_epoch_222_3)

	#add duration and daily
	prepath = '../data/'

	duration_222_1 = pd.read_csv(prepath + 'NY222_3760120_2015-08-11_to_2016-08-11_20180809165417.csv', skiprows=3)
	duration_222_2 = pd.read_csv(prepath + 'NY222_3760120_2016-08-11_to_2017-08-11_20180809003645.csv', skiprows=3)
	duration_222_3 = pd.read_csv(prepath + 'NY222_3760120_2017-08-11_to_2018-08-08_20180809004150.csv', skiprows=3)
	duration_231_1 = pd.read_csv(prepath + 'NY231_4086548_2016-07-05_to_2017-07-05_20180809005536.csv', skiprows=3)
	duration_231_2 = pd.read_csv(prepath + 'NY231_4086548_2017-07-05_to_2018-08-08_20180809010451.csv', skiprows=3)

	duration_222 = pd.concat([duration_222_1, duration_222_2, duration_222_3])
	duration_231 = pd.concat([duration_231_1, duration_231_2])

	daily_222 = pd.read_csv(prepath + 'NY222_2015-08-11_to_2018-06-12_daily_20180613153105.csv', skiprows=3)
	daily_231 = pd.read_csv(prepath + 'NY231_2016-07-05_to_2018-06-12_daily_20180613153815.csv', skiprows=3)
	p222.add_duration(prep.prep_duration(duration_222))
	p231.add_duration(prep.prep_duration(duration_231))
	p231.add_daily(prep.prep_daily(daily_231))
	return p231, p222_1, p222_2, p222_3