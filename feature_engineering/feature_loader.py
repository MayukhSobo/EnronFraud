import pickle
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from feature_format import feature_format, target_feature_split

DATA_PICKLE_SOURCE = '../data/final_project_dataset.pkl'


class Features:
    __metaclass__ = ABCMeta
    targetCol = None
    featureCols = None
    '''
    This class extracts all the features
    from the feature dump and then creates
    dataframes for the respective ML operations
    '''

    def __init__(self, target):
        with open(DATA_PICKLE_SOURCE, 'r') as data_file:
            self.data_dict = pickle.load(data_file)
        Features.targetCol = target
        self.Ids = self.data_dict.keys()
        Features.featureCols = self.data_dict[self.Ids[0]].keys()
        # print self.featureCols

        # Removing the outliers
        self.data_dict.pop('TOTAL')

    # self._dataframe = pd.DataFrame.from_records(self.data_dict.values(),
    # 	index=self.data_dict.keys())

    @staticmethod
    def prepare_features(rCols='email_address'):
        # Removing all the columns that are not required
        if not isinstance(rCols, list):
            rCols = [rCols]
        for each in rCols:
            Features.featureCols.remove(each)
        # Removing the target column
        Features.featureCols.remove(Features.targetCol)

    # @property
    # def dataframe(self):
    # 	return self._dataframe

    @abstractmethod
    def feature_splits(self):
        pass


class FeatureExtract(Features):
    def __init__(self, testSize=0.3, randomState=42, target='poi', featureList="*"):
        super(self.__class__, self).__init__(target)
        FeatureExtract.prepare_features()

        self.rs = randomState
        self.ts = testSize
        self.df_test = None
        self.df_train = None
        labels, features = self._parse_features(featureList)
        self.df = pd.DataFrame(features, columns=self.featureList[1::])
        self.df[FeatureExtract.targetCol] = np.array(labels, dtype=int)
        # self.feature_splits()
        # self.df = self.df_train.append(self.df_test)

    def feature_splits(self):
        x_train, x_test, y_train, y_test = train_test_split(np.array(self.df[FeatureExtract.featureCols]),
                                                            np.array(self.df[FeatureExtract.targetCol]),
                                                            test_size=self.ts,
                                                            random_state=self.rs)
        self.df_train = pd.DataFrame(x_train, columns=FeatureExtract.featureCols)
        self.df_train[self.targetCol] = np.array(y_train, dtype=int)
        self.df_test = pd.DataFrame(x_test, columns=FeatureExtract.featureCols)
        self.df_test[self.targetCol] = np.array(y_test, dtype=int)

    def _parse_features(self, featureList):
        if featureList == '*':
            self.featureList = [FeatureExtract.targetCol] + FeatureExtract.featureCols
        else:
            if FeatureExtract.targetCol not in featureList:
                self.featureList = [FeatureExtract.targetCol] + featureList
        # print featureList
        data = feature_format(self.data_dict, self.featureList, sort_keys=True)
        return target_feature_split(data)

    @property
    def train(self):
        return self.df_train

    @property
    def test(self):
        return self.df_test

# if __name__ == '__main__':
# 	f = FeatureExtract()
# 	# print f.train.head(n=1)
