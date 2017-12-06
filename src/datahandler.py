#
# DataHandler - Load data from the space delimited data set. There are a
#               number of different labels, but AdaBoost needs -1/+1 labels.
#               Some extra functionality is added to specify the labels in 
#               which you are interested.
#
import numpy as np

class DataHandler(object):

	def __init__(self, training_data, test_data):
		self._train = self.load_from_file(training_data)
		self._test  = self.load_from_file(test_data)

	def load_from_file(self, filepath):
		# Use genfromtxt() instead of loadtxt() because it handles
		# trailing spaces better
		print("Importing data from %s" % filepath)
		return np.genfromtxt(filepath, delimiter=' ')

	def get_training_data(self):
		return self._train

	def get_test_data(self):
		return self._test

	def get_training_features(self, y_pos, y_neg):
		if np.size(self._train, 1) > 1:
			# Grab all but the labels in the first column
			labels = self._train[:,0]
			data = self._train[:,1:]

			rows_pos = np.where(labels == y_pos)
			rows_neg = np.where(labels == y_neg)
			rows = np.append(rows_pos, rows_neg)
			return self._train[rows, 1:]

	def get_test_features(self, y_pos, y_neg):
		if np.size(self._test, 1) > 1:
			# Grab all but the labels in the first column
			labels = self._test[:,0]
			data = self._test[:,1:]

			rows_pos = np.where(labels == y_pos)
			rows_neg = np.where(labels == y_neg)
			rows = np.append(rows_pos, rows_neg)
			return self._test[rows, 1:]

	def get_training_labels(self, y_pos, y_neg):
		if np.size(self._train, 1) > 0:
			# Labels sit in the first column. We only want to grab the positive
			# negative labels in which we are interested
			labels = self._train[:, 0]
			rows_pos = np.where(labels == y_pos)
			rows_neg = np.where(labels == y_neg)
			rows = np.append(rows_pos, rows_neg)
			# Convert [0-9] labels to {-1, +1}
			labels = labels[rows]
			labels[labels == y_pos] = 1
			labels[labels == y_neg] = -1
			return labels

	def get_test_labels(self, y_pos, y_neg):
		if np.size(self._test, 1) > 0:
			# Labels sit in the first column
			labels = self._test[:, 0]
			rows_pos = np.where(labels == y_pos)
			rows_neg = np.where(labels == y_neg)
			rows = np.append(rows_pos, rows_neg)
			# Convert [0-9] labels to {-1, +1}
			labels = labels[rows]
			labels[labels == y_pos] = 1
			labels[labels == y_neg] = -1
			return labels
