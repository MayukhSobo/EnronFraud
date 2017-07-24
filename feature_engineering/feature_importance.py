# import pandas as pd
from feature_loader import FeatureExtract


class Importance:
	def __init__(self, algo, save=True, fileName='importance.jpg'):
		if algo.lower() not in ['xgboost', 'random_forest']:
			raise NotImplementedError("Algorithm support not implemented")
		# // TODO implementation for save = False
		_data = FeatureExtract()
		self.test_data = _data.test
		self.train_data = _data.train
		self.data = self.train_data.append(self.test_data)

	def get_importance_xgboost(self, cv=True):
		pass

	def get_importance_rf(self, cv=True):
		pass

	def K_Best(self, k, eval_func):
		pass


if __name__ == '__main__':
	s = Importance(algo='XGBoost')
