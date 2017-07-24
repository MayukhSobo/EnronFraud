import pickle
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from feature_format import featureFormat, targetFeatureSplit

DATA_PICKLE_SOURCE = '../data/final_project_dataset.pkl'


class Features:

	__metaclass__ = ABCMeta
	'''
	This class extracts all the features
	from the feature dump and then creates
	dataframes for the respective ML operations
	'''
	def __init__(self, target):
		with open(DATA_PICKLE_SOURCE, 'r') as data_file:
			self.data_dict = pickle.load(data_file)
		self.targetCol = target
		self.Ids = self.data_dict.keys()
		self.featureCols = self.data_dict[self.Ids[0]].keys()
		self.data_dict.pop('TOTAL')
		# self._dataframe = pd.DataFrame.from_records(self.data_dict.values(),
		# 	index=self.data_dict.keys())

	def prepare_features(self, rCols=['email_address']):
		for each in rCols:
			self.featureCols.remove(each)
		self.featureCols.remove(self.targetCol)

	# @property
	# def dataframe(self):
	# 	return self._dataframe

	@abstractmethod
	def feature_splits(self, labels, features):
		pass


class FeatureExtract(Features):
	def __init__(self, testSize=0.3, randomState=42, target='poi', featureList="*"):
		super(self.__class__, self).__init__(target)
		self.prepare_features()
		self.rs = randomState
		self.ts = testSize
		labels, features = self._parse_features(featureList)
		# print len(labels)
		self.feature_splits(labels, features)

	def feature_splits(self, labels, features):
		X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.ts, random_state=self.rs)
		self.df_train = pd.DataFrame(X_train, columns=self.featureList[1::])
		self.df_train[self.targetCol] = np.array(y_train, dtype=int)
		self.df_test = pd.DataFrame(X_test, columns=self.featureList[1::])
		self.df_test[self.targetCol] = np.array(y_test, dtype=int)

	def _parse_features(self, featureList):
		if featureList == '*':
			self.featureList = [self.targetCol] + self.featureCols
		else:
			if self.targetCol not in featureList:
				self.featureList = [self.targetCol] + featureList
		# print featureList
		data = featureFormat(self.data_dict, self.featureList, sort_keys=True)
		return targetFeatureSplit(data)

	@property
	def train(self):
		return self.df_train

	@property
	def test(self):
		return self.df_test


# if __name__ == '__main__':
# 	f = FeatureExtract()
# 	# print f.train.head(n=1)
