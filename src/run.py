#!/usr/bin/env python

from adaboost import AdaBoost

ada = AdaBoost("./zip.train",
			   "./zip.test")
ada.run()
