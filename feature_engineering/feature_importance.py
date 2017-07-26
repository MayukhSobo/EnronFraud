import xgboost as xgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from feature_loader import FeatureExtract
import operator
from collections import OrderedDict


class Importance:

    def __init__(self, algo):
        if algo.lower() not in ['xgboost', 'random_forest']:
            raise NotImplementedError("Algorithm support not implemented")
        # // TODO implementation for save = False
        _data = FeatureExtract()
        self.test_data = _data.test
        self.train_data = _data.train
        self.data = self.train_data.append(self.test_data)
        self.features = self.train_data.drop(['poi'], axis=1).columns.values

    def get_importance_xgboost(self, file_path=None, save=True, cv=False):
        """
        Function crates plot of feature importances
        using xgboost model if save=True. It also can
        perform parameter tuning if cv=True.

        :param file_path: image file to save
        :param save: True for saving as image
        :param cv: True for XGBoost's param tuning
        :return: None for save=True, list of features inf save=False
        """

        # Default XGB parameters
        xgb_params = {"objective": "binary:logistic",
                      "eta": 0.01,
                      "max_depth": 8,
                      'colsample_bytree': 0.8,
                      'subsample': 0.8,
                      "seed": 42,
                      "silent": 1,
                      'n_estimators': 100,
                      'gamma': 0,
                      'early_stopping_rounds': 900,
                      'eval_metric': 'error@0.7'
                      }
        num_boost_round = 100
        X_train = self.train_data.drop(['poi'], axis=1)
        y_train = self.train_data.loc[:, 'poi'].values
        if not cv:
            # don't perform parameter tuning
            dtrain = xgb.DMatrix(X_train, y_train, feature_names=self.features)
            gbdt = xgb.train(xgb_params, dtrain, num_boost_round)
            importance = sorted(gbdt.get_fscore().iteritems(), key=operator.itemgetter(1), reverse=True)
            if not save:
                return importance
            else:
                import pandas as pd
                from matplotlib import pylab as plt
                df = pd.DataFrame(importance, columns=['feature', 'fscore'])
                df['fscore'] = df['fscore'] / df['fscore'].sum()
                plt.figure()
                df.plot()
                df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 10))
                plt.title('XGBoost Feature Importance')
                plt.xlabel('relative importance')
                plt.gcf().savefig(file_path)

    def get_importance_rf(self, file_path=None, save=True, cv=False):
        X_train = self.train_data.drop(['poi'], axis=1)
        X_train = np.array(X_train)
        y_train = self.train_data.loc[:, 'poi'].values
        forest = ExtraTreesClassifier(n_estimators=250, random_state=42)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        # indices = np.argsort(importances)[::-1]
        ftr_imps = OrderedDict()
        for each in list(np.argsort(importances)[::-1]):
            ftr_imps[self.features[each]] = importances[each]
        if not save:
            return ftr_imps
        else:
            from matplotlib import pylab as plt
            plt.figure(figsize=(17, 10))
            plt.title('Random Forest Feature Importance')
            plt.barh(range(X_train.shape[1]), ftr_imps.values(), color="r")
            plt.yticks(range(X_train.shape[1]), ftr_imps.keys())
            plt.xlim([0, 0.3])
            plt.xlabel('relative importance')
            plt.gcf().savefig(file_path)


    def K_Best(self, k, eval_func):
        pass


if __name__ == '__main__':
    imp = Importance(algo='XGBoost')
    # imp.get_importance_xgboost(file_path='feature_importance_xgboost.png', save=True)
    imp.get_importance_rf(file_path='feature_importance_rf.png', save=True)
