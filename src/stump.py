import sys
import numpy as np


class Stump(object):
	def __init__(self, info_gain, dimension, split_value):
		self._info_gain = info_gain
		self._dimension = dimension
		self._split_value = split_value

	def is_info_gain_better(self, info_gain):
		if info_gain > self._info_gain:
			return True
		return False

	def update_info_gain(self, info_gain, dimension, split_value, X, y, w):
		self._info_gain = info_gain
		self._dimension = dimension
		self._split_value = split_value
		(self._left_sign, self._right_sign) = self.find_signs(X, y, w)

	def find_signs(self, X, y, w):
		y_left = y[np.where(X[:,self._dimension] <= self._split_value)]
		w_left = w[np.where(X[:,self._dimension] <= self._split_value)]
		neg_left_sum = np.sum(w_left[np.where(y_left == -1)])
		pos_left_sum = np.sum(w_left[np.where(y_left == 1)])
		w_left_sum = pos_left_sum - neg_left_sum

		y_right = y[np.where(X[:,self._dimension] > self._split_value)]
		w_right = w[np.where(X[:,self._dimension] > self._split_value)]
		neg_right_sum = np.sum(w_right[np.where(y_right == -1)])
		pos_right_sum = np.sum(w_right[np.where(y_right == 1)])
		w_right_sum = pos_right_sum - neg_right_sum

		return (np.sign(w_left_sum), np.sign(w_right_sum))

	def calculate_error(self, X, y, w):
		n = np.size(X, 0)

		predictions = np.zeros(n)
		error = 0
		for i in xrange(n):
			if X[i,self._dimension] <= self._split_value:
				predictions[i] = self._left_sign
			else:
				predictions[i] = self._right_sign

			if predictions[i] != y[i]:
				error += w[i]

		return (error, predictions)

