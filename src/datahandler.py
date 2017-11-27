import numpy as np

class DataHandler(object):
	def __init__(self, training_data, test_data):
		self._train = self.load_from_file(training_data)
		self._test  = self.load_from_file(test_data)

	def load_from_file(self, filepath):
		# Use genfromtxt() instead of loadtxt() because it handles
		# trailing spaces better
		print("Importing data from %s" % filepath)
		out = np.genfromtxt(filepath, delimiter=' ')
		print("Import successful")

	def get_training_data(self):
		return self._train

	def get_test_data(self):
		return self._test

	def get_training_features(self):
		if np.size(self._train, 1) > 1:
			# Grab all but the labels in the first column
			return self._train[:, 1:]

	def get_test_features(self):
		if np.size(self._test, 1) > 1:
			# Grab all but the labels in the first column
			return self._test[:, 1:]

	def get_training_labels(self):
		if np.size(self._train, 1) > 0:
			# Labels sit in the first column
			return self._train[:, 0]

	def get_test_labels(self):
		if np.size(self._test, 1) > 0:
			# Labels sit in the first column
			return self._test[:, 0]
			