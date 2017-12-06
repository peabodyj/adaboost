#
# Stump - Used in AdaBoost as a weak learner. Splits data based on a single
#         feature and split value. AKA not quite a tree. One side of the
#         split will predict a positive label and one side will predict a
#         negative label.
#
import numpy as np


class Stump(object):

	def __init__(self, info_gain, dimension, split_value):
		# Save the info gain level to help us find the best split easier
		self._info_gain = info_gain
		# The details of our weak learner
		self._dimension = dimension
		self._split_value = split_value

	def update_info_gain_if_better(self, info_gain, dimension, split_value, X, y, w):
		# Use this function to grab the best stump possible
		if info_gain > self._info_gain:
			self._info_gain = info_gain
			self._dimension = dimension
			self._split_value = split_value
			(self._left_sign, self._right_sign) = self.find_signs(X, y, w)

	def find_signs(self, X, y, w):
		# "Left" and "Right" just signify each side of our split
		# Find what the majority label is for each side to determine what we
		# will predict
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

		# Return as +1/0/-1 value to make error calclation much easier later
		return (np.sign(w_left_sum), np.sign(w_right_sum))

	def calculate_error(self, X, y, w):
		n = np.size(X, 0)

		predictions = np.zeros(n)
		error = 0
		# Keep a running total for error for each mispredicted label.
		# TODO: Can probably numpy.vectorize() this if we need slight speed ups
		for i in xrange(n):
			if X[i,self._dimension] <= self._split_value:
				predictions[i] = self._left_sign
			else:
				predictions[i] = self._right_sign

			if predictions[i] != y[i]:
				error += w[i]

		return (error, predictions)

