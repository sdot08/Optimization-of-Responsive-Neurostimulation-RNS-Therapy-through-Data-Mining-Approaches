class patient():
	def __init__(self, id):
		self.id = id
		self.duration = None
		self.daily = None
		self.hasduration = False
		self.hasdaily = False
		pass

	def add_duration(self,dat):
		self.duration = dat
	def add_epochinfo(self, start, end, num_per_epoch):
		self.epoch_info = {}
		self.epoch_info['start'] = start
		self.epoch_info['end'] = end
		self.epoch_info['num_per_epoch'] = num_per_epoch

	def add_daily(self,dat):
		self.daily = dat
