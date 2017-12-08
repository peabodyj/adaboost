#!/usr/bin/env python

import sys
from adaboost import AdaBoost

# defaults
stumps = 50
train = "../data/zip.train"
test = "../data/zip.test"
pos_lbl = 1
neg_lbl = 2

if len(sys.argv) > 5:
	stumps = int(sys.argv[1])
	train = sys.argv[2]
	test = sys.argv[3]
	pos_lbl = int(sys.argv[4])
	neg_lbl = int(sys.argv[5])
else:
    print("Not enough arguments, using defaults.")


ada = AdaBoost(stumps, train, test, pos_lbl, neg_lbl)
ada.run()
