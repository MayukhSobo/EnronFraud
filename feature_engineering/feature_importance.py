import xgboost as xgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from feature_loader import FeatureExtract
import operator
from collections import OrderedDict


class Importance:

    def __init__(self, algo):
        if algo.lower() not in ['xgboost', 'random_forest', 'k_best', '*']:
            raise NotImplementedError("Algorithm support not implemented")
        # // TODO implementation for save = False
        _data = FeatureExtract()
        self.test_data = _data.test
        self.train_data = _data.train
        self.data = self.train_data.append(self.test_data)
        self.features = self.train_data.drop(['poi'], axis=1).columns.values

    def get_importance_xgboost(self, file_path=None, save=True, cv=False, k=5):
        """
        Function crates plot of feature importance
        using xgboost model if save=True. It also can
        perform parameter tuning if cv=True.

        :param k: Number of features
        :param file_path: image file to save
        :param save: True for saving as image
        :param cv: True for XGBoost's param tuning
        :return: None for save=True, list of features inf save=False
        """

        if k < 1 or k > len(self.features):
            raise ValueError('Error in k value of {}'.format(k))

        # Default XGB parameters

        num_boost_round = 100
        X_train = self.train_data.drop(['poi'], axis=1)
        y_train = self.train_data.loc[:, 'poi'].values
        xgb_params = {
            "objective": "binary:logistic",
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
        if not cv:
            # don't perform parameter tuning
            dtrain = xgb.DMatrix(X_train, y_train, feature_names=self.features)
            gbdt = xgb.train(xgb_params, dtrain, num_boost_round)
            importance = sorted(gbdt.get_fscore().iteritems(), key=operator.itemgetter(1), reverse=True)
        else:
            from sklearn.model_selection import GridSearchCV
            # If cross validation is applied
            xgb_params = {
                "objective": "binary:logistic",
                "learning_rate": 0.01,
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
            parameters = {
                'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
                'max_depth': [3, 4, 5, 8, 10],
                'subsample': [0.5, 0.8, 0.9, 1.0],
                'colsample_bytree': list(np.arange(0.01, 0.1, 0.01))
                # 'gamma': range(10, 16)
            }
            _clf = xgb.XGBClassifier(xgb_params, num_boost_round=100)
            clf = GridSearchCV(_clf, parameters, n_jobs=4, cv=7, scoring='average_precision')
            clf.fit(np.array(X_train), y_train)
            best_learning_rate = clf.best_estimator_.learning_rate
            best_max_depth = clf.best_estimator_.max_depth
            best_subsample = clf.best_estimator_.subsample
            xgb_params['learning_rate'] = best_learning_rate
            xgb_params['max_depth'] = best_max_depth
            xgb_params['subsample'] = best_subsample
            dtrain = xgb.DMatrix(X_train, y_train, feature_names=self.features)
            gbdt = xgb.train(xgb_params, dtrain, num_boost_round=100)
            importance = sorted(gbdt.get_fscore().iteritems(), key=operator.itemgetter(1), reverse=True)
        if not save:
            return OrderedDict(importance[:k])
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

    def get_importance_rf(self, file_path=None, save=True, cv=False, k=5):
        X_train = self.train_data.drop(['poi'], axis=1)
        X_train = np.array(X_train)
        y_train = self.train_data.loc[:, 'poi'].values
        forest = ExtraTreesClassifier(n_estimators=250, random_state=42)
        forest.fit(X_train, y_train)
        importance = forest.feature_importances_
        ftr_imps = OrderedDict()
        for each in list(np.argsort(importance)[::-1]):
            ftr_imps[self.features[each]] = importance[each]
        if not save:
            return OrderedDict(list(ftr_imps.items())[:k])
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
        from sklearn.feature_selection import SelectKBest
        if k < 1:
            raise ValueError('K value error in K_Best.')

        # We can support more eval functions here
        if eval_func.lower() not in ['classif']:
            raise NotImplementedError('The eval function {} is not implemented'.format(eval_func))

        if eval_func.lower() == 'classif':
            from sklearn.feature_selection import f_classif
            select_k_best = SelectKBest(f_classif, k)
            train_features = np.array(self.train_data.drop(['poi'], axis=1))
            train_labels = self.train_data.loc[:, 'poi'].values
            select_k_best.fit(train_features, train_labels)
            best_features = self.features[select_k_best.get_support(indices=True)]
            best_scores = select_k_best.scores_[select_k_best.get_support(indices=True)]
            top_k_features = dict(zip(best_features, best_scores))
            return OrderedDict(sorted(top_k_features.iteritems(), key=operator.itemgetter(1), reverse=True))

if __name__ == '__main__':
    imp = Importance(algo='*')
    # print '\t\t\t' + '-' * 5 + ' Top 5 features ' + '-' * 5
    # print '\n\tK_best algorithm'
    # print imp.K_Best(5, 'classif').keys()
    # print '\n\tXGBoost algorithm'
    print imp.get_importance_xgboost(save=False, cv=True).keys()
    print imp.get_importance_xgboost(save=False).keys()
    # print '\n\tRandom Forest algorithm'
    # print imp.get_importance_rf(save=False).keys()
