from hyperparams import Hyperparams as hp
from patient import patient

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
	return p231, p222_1, p222_2, p222_3