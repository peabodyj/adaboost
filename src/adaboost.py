import numpy as np
from datahandler import DataHandler

class AdaBoost:
	def __init__(self, training_data, test_data):
		self.data_handler = DataHandler(training_data, test_data)

	def run(self):
		print("Run.")