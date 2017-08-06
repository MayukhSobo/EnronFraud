from copy import deepcopy
import pickle
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
# from feature_format import feature_format, target_feature_split

DATA_PICKLE_SOURCE = 'data/final_project_dataset.pkl'


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
        self.data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

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
    def __init__(self, testSize=0.3, randomState=42, target='poi'):
        super(self.__class__, self).__init__(target)
        FeatureExtract.prepare_features()
        self.rs = randomState
        self.ts = testSize
        self.df_test = None
        self.df_train = None
        self.df = pd.DataFrame.from_records(self.data_dict.values(), index=self.data_dict.keys())
        targetColumn = self.df[FeatureExtract.targetCol]
        self.df.drop(FeatureExtract.targetCol, inplace=True, axis=1)
        targetColumn = targetColumn.astype(int)
        self.df[FeatureExtract.targetCol] = targetColumn
        self.df.replace('NaN', 0, inplace=True)
        self.df = self.df[FeatureExtract.featureCols + [FeatureExtract.targetCol]]
        self.orig_df = deepcopy(self.df)
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

    # def _parse_features(self, featureList):
    #     if featureList == '*':
    #         self.featureList = [FeatureExtract.targetCol] + FeatureExtract.featureCols
    #     else:
    #         if FeatureExtract.targetCol not in featureList:
    #             self.featureList = [FeatureExtract.targetCol] + featureList
    #     # print featureList
    #     data = feature_format(self.data_dict, self.featureList, sort_keys=True)
    #     return target_feature_split(data)

    @property
    def train(self):
        return self.df_train

    @property
    def test(self):
        return self.df_test

    # def adhoc_feature_parse(self, columns='*', merge_train_test=False):
    #     """
    #
    #     :param columns: List - Columns that is to be parsed
    #     :param merge_train_test: Bool - Merger the test and train
    #     :return: Numpy array
    #     """
    #     if not isinstance(columns, list):
    #         columns = list(columns)
    #     if merge_train_test:
    #         if columns == ['*']:
    #             return self.df.as_matrix()
    #         else:
    #             from copy import deepcopy
    #             features = deepcopy(columns)
    #             features.append('poi')
    #             return self.df[features].as_matrix()
    #     else:
    #         if columns == ['*']:
    #             return self.train.as_matrix(), self.test.as_matrix()
    #         else:
    #             from copy import deepcopy
    #             features = deepcopy(columns)
    #             features.append('poi')
    #             return self.train[columns].as_matrix(), self.test[columns].as_matrix()

# 	f = FeatureExtract()
# 	# print f.train.head(n=1)
