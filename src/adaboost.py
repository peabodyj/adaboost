#
# AdaBoost - Perform AdaBoost algorithm on a set of training/test data.
#            Try optimizing for time by using numpy functionality and 
#            avoiding "for" loops during calculations.
#
import numpy as np
# TODO: Add error plotting
#import matplotlib.pyplot as plt

from datahandler import DataHandler
from stump import Stump

class AdaBoost(object):

	def __init__(self, num_stumps, training_data, test_data, pos_lbl, neg_lbl):
		self._data_handler = DataHandler(training_data, test_data)
		# Represents the number of time steps we will look for
		# a new weak learner (or stump)
		self._num_stumps = num_stumps

		# Extract test and training data for the labels we are interested in
		self._X_train = self._data_handler.get_training_features(pos_lbl, neg_lbl)
		self._y_train = self._data_handler.get_training_labels(pos_lbl, neg_lbl)
		self._X_test = self._data_handler.get_test_features(pos_lbl, neg_lbl)
		self._y_test = self._data_handler.get_test_labels(pos_lbl, neg_lbl)

		# Get these ahead of time because we use them a lot
		# Size and dimension of training data
		self._n_train = np.size(self._X_train, 0)
		self._d_train = np.size(self._X_train, 1)
		# Size and dimension of test data
		self._n_test = np.size(self._X_test, 0)
		self._d_test = np.size(self._X_test, 1)

		# We scan across features in our data over and over looking for good
		# midpoints. Since the data does not change during the run, we can calculate
		# once and save the values of the midpoints to cut down on runtime.
		self._midpoints = {}

		# Save for plotting
		# TODO: Add plotting so we can actually see results!
		self._train_errors = []
		self._test_errors = []

	def run(self):
		# Create a normalized weight for each sample (1/n)
		weights = np.full((self._n_train, 1), 1.0 / self._n_train)
		# Used in our calculation later
		alpha = np.zeros(self._num_stumps)

		# Store the best stump for each time step here
		stumps = {}
		# Represent each time step t from 1..T
		for t in xrange(self._num_stumps):
			print("Weak Learner #"+str(t))
			# Finding the new stump/weak learner is the most time consuming
			# part of the alorithm
			stump = self.find_stump(weights)
			stumps[t] = stump

			(eps_t, predictions) = stump.calculate_error(self._X_train, self._y_train, weights)

			# Need to early terminate if epsilon from this timestep is too high
			# because we will be making our prediction worse
			if eps_t > 0.5:
				break

			alpha[t] = 0.5 * np.log2((1 - eps_t) / eps_t)

			# Update weights based on our new information
			for i in xrange(self._n_train):
				weights[i] = weights[i] * np.exp(-1 * alpha[t] * predictions[i] * self._y_train[i])

			# Re-normalize the weights
			tw = np.sum(weights)
			weights = np.divide(weights, tw)


			# The timestep is now complete, lets check our current error so we
			# can create a pretty plot of training error vs test error over time.
			train_preds = self.get_hypothesis(self._X_train, self._y_train, alpha, stumps, weights, t)
			train_err = 0
			for i in xrange(self._n_train):
				if self._y_train[i] != train_preds[i]:
					train_err += 1

			train_err = float(train_err) / self._n_train
			self._train_errors.append(train_err)

			test_preds = self.get_hypothesis(self._X_test, self._y_test, alpha, stumps, weights, t)
			test_err = 0
			for i in xrange(self._n_test):
				if self._y_test[i] != test_preds[i]:
					test_err += 1

			test_err = float(test_err) / self._n_test
			self._test_errors.append(test_err)

	# Find the stump with the best feature / value split
	def find_stump(self, weights):
		# The stump we are looking for is the split in the dimension's data where we
		# have the highest info gain. We will only try splitting at the midpoints between
		# any two training points so we do not exhausively try every location.
		best_stump = Stump(-1, -1, -1)
		for dimension in xrange(self._d_train):
			# Since we always have the same midpoints for each dimension,
			# we can save a little run time in exchange for more memory by
			# saving the midpoints
			if dimension not in self._midpoints.keys():
				x_sorted = np.sort(self._X_train[dimension,:])
				x_sorted = np.unique(x_sorted)

				# numpy trick to make an array of midpoints between each pair of adjacent values
				midpoints = x_sorted[:-1] + np.diff(x_sorted) / 2
				self._midpoints[dimension] = midpoints

			info_gains = [self.check_stump_info_gain(dimension, p, weights) for p in self._midpoints[dimension]]
			info_gains = np.array(info_gains)

			# We need to keep track of our current best
			max_ig = np.max(info_gains)
			max_idx = np.argmax(info_gains)
			best_stump.update_info_gain_if_better(max_ig, dimension, self._midpoints[dimension][max_idx], self._X_train, self._y_train, weights)

		return best_stump


	def check_stump_info_gain(self, dimension, split, weights):
		# Nomenclature: in this class "left" will be one side of the split value
		# and "right" will be on the other side
		parent_entropy = self.calculate_entropy(self._X_train, self._y_train, weights)

		# Grab total weights of values on either side of the chosen stump split value
		left_idx = np.where(self._X_train[:,dimension] <= split)
		left_weights_tot = np.sum(weights[left_idx])
		right_idx = np.where(self._X_train[:,dimension] > split)
		right_weights_tot = np.sum(weights[right_idx])

		total_w = np.sum(weights)

		left_entropy = 0
		if left_weights_tot != 0:
			left_entropy = self.calculate_entropy(self._X_train[left_idx,:], self._y_train[left_idx], weights[left_idx])

		right_entropy = 0
		if right_weights_tot != 0:
			right_entropy = self.calculate_entropy(self._X_train[right_idx,:], self._y_train[right_idx], weights[right_idx])

		# Child entropy becomes weighted based on left and right sides of the split
		child_entropy = (left_weights_tot / total_w) * left_entropy + (right_weights_tot / total_w) * right_entropy
		# Info gain is the difference between the parent and child entropy, aka the
		# how much better we got by adding this new stump
		info_gain = np.absolute(parent_entropy - child_entropy)

		return info_gain

	def calculate_entropy(self, X, y, w):
		n = np.size(X, 0)
		d = np.size(X, 1)

		total_w = np.sum(w)

		# Sum the weights of the matching labels (positive / negative)
		neg_w = w[np.where(y == -1)]
		neg_w = np.sum(neg_w)
		pos_w = w[np.where(y == 1)]
		pos_w = np.sum(pos_w)

		# We do not want to include the entropy if we get log(0) on one of the
		# terms, so we do it in steps.
		entropy = 0
		if neg_w != 0:
			entropy -= (neg_w / total_w) * np.log2(neg_w / total_w)
		if pos_w != 0:
			entropy -= (pos_w / total_w) * np.log2(pos_w / total_w)

		return entropy

	# Gets data predictions based on our current group of weak learners
	def get_hypothesis(self, X, y, alpha, stumps, weights, timestep):
		n = np.size(X, 0)
		predictions_sum = np.zeros(n)
		for t in xrange(timestep+1):
			(error, predictions) = stumps[t].calculate_error(X, y, weights)
			predictions_sum = np.add(predictions_sum, predictions)

		return np.sign(predictions_sum)
