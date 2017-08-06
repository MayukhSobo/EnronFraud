from feature_engineering import feature_importance
from feature_engineering import feature_loader
from feature_engineering import feature_misc
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier, dump_classifier_and_data
import timeit

# ------------ Creating a Feature Object ------------ #
f = feature_loader.FeatureExtract()
# ------------ Creating a Axillary  feature operation Object ---------- #
aux = feature_misc.Aux('create', f)
#  ------------------ Create a new features ------------------ #
'''
Feature Name        : poi_interaction
Feature Expression  : poi_interaction = (from_this_person_to_poi / from_messages) +
                                        (from_poi_to_this_person / to_messages)

Description         : As both ```from_this_person_to_poi``` and ```from_poi_to_this_person```
                      represents how many emails the person received or sent to the POI, it is
                      important to keep a track of the fact that how much this person is active with
                      the POIs. Chances are that if a person is interacting with POIs too much,
                      they themselves are POI. Hence the following count can be very important to
                      determine if they are POI or not. One thing to notice is that we have normalised
                      the from_this_person_to_poi & from_poi_to_this_person
                      by total number of from and to messages. This makes it even more robust.
'''
aux.operate('/', 'from_this_person_to_poi_stand', features=['from_this_person_to_poi', 'from_messages'])
aux.operate('/', 'from_poi_to_this_person_stand', features=['from_poi_to_this_person', 'to_messages'])
aux.operate('+', 'poi_interaction', features=['from_poi_to_this_person_stand', 'from_this_person_to_poi_stand'])

'''
Feature Name        : income_ratio
Feature Expression  : (salary + bonus + long_term_incentive) / total_payment

Description         : This is very powerful feature because it standardise the income of an employee
                      with the ```salary``` and ```bonus```. The point is that is the person is not a
                      POI then, this ratio wouldn't be too large but if a person is POI, this value
                      would probably be quite large.
'''
aux.operate('+', 'bonusPlusSalaryPlusIncentives', features=['bonus', 'salary', 'long_term_incentive'])
aux.operate(sign='/', new_feature='income_ratio',  remove=False, features=['bonusPlusSalaryPlusIncentives',
                                                                           'total_payments'])
aux.operate('/', 'expenses_std', features=['expenses', 'total_payments'], remove=False)
aux.operate('/', 'deferral_payments_std', features=['deferral_payments', 'total_payments'], remove=False)
aux.operate('/', 'other_std', features=['other', 'total_payments'])
feature_loader.FeatureExtract.featureCols.remove('expenses')
feature_loader.FeatureExtract.featureCols.remove('bonusPlusSalaryPlusIncentives')
feature_loader.FeatureExtract.featureCols.remove('deferral_payments')
# ------------ Remove the features that are not required -------------- #
# This feature is kept manual because it is not mandatory
f.df.drop(['from_this_person_to_poi',
           'from_messages',
           'from_poi_to_this_person',
           'to_messages',
           'bonusPlusSalaryPlusIncentives',
           'bonus',
           'salary',
           'long_term_incentive',
           'from_this_person_to_poi_stand',
           'from_poi_to_this_person_stand',
           'expenses',
           'other',
           'total_payments',
           'deferral_payments'],
          axis=1, inplace=True)
#  ------------ Split the dataset for train and test --------- #
f.feature_splits()
# -------------- Feature Selection ---------------- #
#
imp = feature_importance.Importance(algo='*', fObj=f)
# print imp.get_importance_rf(save=True, file_path='importance_random_forest.png')
# print imp.get_importance_xgboost(save=True, file_path='importance_xgboost.png')
# print imp.get_importance_kBest(k=5, eval_func='classif').keys()

f.df.to_pickle('final_df.pkl')

# ~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~ #

# Model 1

'''
    Feature Selection:
        algorithm: Random Forest
        n_estimator: 250
        random_state: 42
        number of features: 5
        cross validation: False

    Feature Scaling:
        None

    Cross Validation:
        None

    Classification:
        algo: Gaussian Naive Bayes
'''

# important_features_rf = imp.get_importance_rf(save=False).keys()
# dataset_rf = f.adhoc_feature_parse(columns=important_features_rf, merge_train_test=True)
# clf = GaussianNB()
# start = timeit.timeit()
# test_classifier(clf, dataset_rf)
# print "Elapsed: " + str(timeit.timeit() - start)

# Model 2

'''
    Feature Selection:
        algorithm: XGBoost
        early_stopping_rounds: 900
        num_boosting_rounds: 100
        eval_metric: error@0.7
        objective: binary:logistic
        random_state: 42
        learning_rate: 0.01
        max_depth: 8
        subsample: 0.8
        colsample_bytree: 0.8
        cross validation: False

    Feature Scaling:
        None

    Cross Validation:
        None

    Classification:
        algo: AdaBoostClassifier
'''

# important_features_xgb = imp.get_importance_xgboost(save=False).keys()
# dataset_xgb = f.adhoc_feature_parse(columns=important_features_xgb, merge_train_test=True)
# clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10), random_state=42)
# start = timeit.timeit()
# test_classifier(clf, dataset_xgb)
# print "Elapsed: " + str(timeit.timeit() - start)

# Model 3

'''
    Feature Selection:
        algorithm: XGBoost
        early_stopping_rounds: 900
        num_boosting_rounds: 100
        eval_metric: error@0.7
        objective: binary:logistic
        random_state: 42
        learning_rate: best from CV
        max_depth: best from CV
        subsample: best from CV
        colsample_bytree: best from CV
        cross validation: True

    Feature Scaling:
        None

    Cross Validation:
        None

    Classification:
        algo: NearestCentroid
'''

# important_features_xgb_cv = imp.get_importance_xgboost(save=False, cv=True).keys()
# dataset_xgb_cv = f.adhoc_feature_parse(columns=important_features_xgb_cv, merge_train_test=True)
# clf = NearestCentroid(shrink_threshold=0.1)
# start = timeit.timeit()
# test_classifier(clf, dataset_xgb_cv)
# print "Elapsed: " + str(timeit.timeit() - start)

# Model 4

'''
    Feature Selection:
        algo: K Best
        k: 5
        eval_func: f_classif

    Feature Scaling:
        None

    Cross Validation:
        None

    Classification:
        algo: SVC
'''

# important_features_kbest = imp.get_importance_kBest(k=5, eval_func='classif').keys()
# dataset_kbest = f.adhoc_feature_parse(columns=important_features_kbest, merge_train_test=True)
# clf = SVC(C=1000, kernel='rbf', gamma=1)
# start = timeit.timeit()
# test_classifier(clf, dataset_kbest)
# print "Elapsed: " + str(timeit.timeit() - start)

# Model 5

'''
    Feature Selection:
        algorithm: XGBoost
        early_stopping_rounds: 900
        num_boosting_rounds: 100
        eval_metric: error@0.7
        objective: binary:logistic
        random_state: 42
        learning_rate: best from CV
        max_depth: best from CV
        subsample: best from CV
        colsample_bytree: best from CV
        cross validation: True

    Feature Scaling:
        None

    Cross Validation:
        None

    Classification:
        algo: Gradient Boosting
'''

# start = timeit.timeit()
# clf = GradientBoostingClassifier()
# test_classifier(clf, dataset_xgb_cv)
# print "Elapsed: " + str(timeit.timeit() - start)


# ------------- Fine tune Classifiers ------------- #

# >>>>>>>>>  SVC with penalty for balanced score <<<<<<<< #
'''
Fine tuned Classifier 1: SVC - with penalty of 1/3.3
                          for biased class. SVC expects
                          the class should be balanced but
                          in reality the class is not. Hence
                          we put the penalty term. So now one
                          class instance of 1 is equal to 3.3
                          instances of class 0

Classifier Result:        Balanced result for balanced precision
                          and accuracy. The dataset is not so good
                          and best balanced score can not be more
                          than 0.4.

Classifier Scores:        Accuracy: 0.83920
                          Precision: 0.38669
                          Recall: 0.35150
                          F1: 0.36826
                          F2: 0.35802
'''
# pipe = Pipeline([('scale', MaxAbsScaler()),
#                  ('reduce_dim', PCA(random_state=42)),
#                  ('classify', SVC(class_weight={0.: 1, 1.: 3.3}))])
#
# number_of_features = range(2, f.df.shape[1] - 1)
#
# C_param = [0.1, 1, 10]
# gamma_param = range(10, 30)
# param_grid = [
#     {
#         'scale': [None, MaxAbsScaler()],
#         'reduce_dim': [PCA(random_state=42)],
#         'reduce_dim__n_components': number_of_features,
#         'classify__C': C_param,
#         'classify__gamma': gamma_param
#     },
#     {
#         'scale': [None, MaxAbsScaler()],
#         'reduce_dim': [SelectKBest()],
#         'reduce_dim__k': number_of_features,
#         'classify__C': C_param,
#         'classify__gamma': gamma_param
#     },
# ]
#
# cv = StratifiedShuffleSplit(random_state=42)
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)
# features = f.df.as_matrix()[:, 0:-2]
# labels = f.df.as_matrix()[:, -1]
# grid.fit(features, labels)
# test_classifier(grid.best_estimator_, f.df.as_matrix())
#
# # >>>>>>>>>>>>>> A better balanced score with Naive Bayes with PCA <<<<<<<<< #
'''
Fine tuned Classifier 2: GaussianNB - This assumes that the
                         features are independent of each other
                         and hence PCA is used to make the dependent
                         components as independent.

Classifier Result:       Because of the PCA, the Naive Bayes performs
                         better than the previous one. We see jump in
                         both Precision and Accuracy than SVC with penalty
                         and hence can comment that Naive Bayes is performing
                         better than the SVC and it's performance is not
                         affected by biased class distribution.

Classifier Scores:        Accuracy: 0.87580
                          Precision: 0.55170
                          Recall: 0.36550
                          F1: 0.43970
                          F2: 0.39196
'''
# pipe = Pipeline([('scale', MaxAbsScaler()),
#                  ('reduce_dim', PCA(random_state=42)),
#                  ('classify', GaussianNB())])
# number_of_components = range(2, f.df.shape[1] - 1)
# param_grid = [
#     {
#         'scale': [None, MaxAbsScaler()],
#         'reduce_dim__n_components': number_of_components
#     }
# ]
# cv = StratifiedShuffleSplit(random_state=42)
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)
# features = f.df.as_matrix()[:, 0:-2]
# labels = f.df.as_matrix()[:, -1]
# grid.fit(features, labels)
# test_classifier(grid.best_estimator_, f.df.as_matrix())

'''
Fine tuned Classifier 3: DecisionTreeClassifier - Decision Tree
                         said to perform better for biased dataset
                         if proper cross validation is done

Classifier Result:       Decision Tree performs best if we tune
                         the max_depth parameter using cross validation
                         parameter. The presort parameter is is used because
                         sorted dataset training time is comparatively lesser

Classifier Scores:        Accuracy: 0.85140
                          Precision: 0.44831
                          Recall: 0.49650
                          F1: 0.47117
                          F2: 0.48605
'''
# pipe = Pipeline([('scale', MaxAbsScaler()),
#                  ('reduce_dim', PCA(random_state=42)),
#                  ('classify', DecisionTreeClassifier(class_weight={0.: 1, 1.: 4}, random_state=42))])
#
# number_of_features = range(2, f.df.shape[1] - 1)
# learning_rate = [0.01, 0.05, 0.3]
# max_depth = range(10, 2, -1)
# presort = [True, False]
# param_grid = [
#     {
#         'scale': [None, MaxAbsScaler()],
#         'reduce_dim': [PCA(random_state=42)],
#         'reduce_dim__n_components': number_of_features,
#         'classify__max_depth': max_depth,
#         'classify__presort': presort
#     },
#     {
#         'scale': [None, MaxAbsScaler()],
#         'reduce_dim': [SelectKBest()],
#         'reduce_dim__k': number_of_features,
#         'classify__max_depth': max_depth,
#         'classify__presort': presort
#     }
# ]
# cv = StratifiedShuffleSplit(random_state=42)
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='precision', n_jobs=-1)
# features = f.df.as_matrix()[:, 0:-2]
# labels = f.df.as_matrix()[:, -1]
# grid.fit(features, labels)
# test_classifier(grid.best_estimator_, f.df.as_matrix())
# >>>>>>>>>>>>>>> The best Result <<<<<<<<<<<<< #

'''
Fine tuned Classifier 3: NearestCentroid - Any KNN or similar
                         algorithms are slow learners and they
                         actually don't need class labels for training
                         to predict. But this dataset is smaller and
                         hence using KNN like algorithms won't be a problem.
                         KNN or similar algorithms are always known for its
                         balanced performance. NearestCentroid is better
                         version of KNN. Instead of just looking at the Neighbours
                         which can an outlier and cause inaccuracy in the classifier,
                         NearestCentroid finds the center of the data and the
                         uses different distance formula to classify the belongings
                         of different points in different region.

Classifier Result:       NearestCentroid give the best result. It not only
                         gives a very goof F2 score but also pulls up the F1
                         score. The enron dataset is not so good but it still
                         gives a good result. Although it brings down the accuracy
                         a little bit than Naive Bayes. But we are not bothered
                         about accuracy though. Note that we are cross-validating
                         with ```precision``` score and not with ```f1``` score.
                         This compromises the recall to get better precision. Even
                         after this recall is not that bad.

Classifier Scores:        Accuracy: 0.82367
                          Precision: 0.40778
                          Recall: 0.71300
                          F1: 0.51883
                          F2: 0.62016
'''

# pipe = Pipeline([('scale', MaxAbsScaler()),
#                  ('reduce_dim', PCA(random_state=42)),
#                  ('classify', NearestCentroid())])
#
# number_of_features = range(2, f.df.shape[1] - 1)
# shrink_threshold = [None, 0.1, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
# param_grid = [
#     {
#         'scale': [None, MaxAbsScaler(), StandardScaler(), MinMaxScaler()],
#         'reduce_dim': [PCA(random_state=42)],
#         'reduce_dim__n_components': number_of_features,
#         'classify__metric': ["euclidean", "manhattan"],
#         'classify__shrink_threshold': shrink_threshold
#     },
#     {
#         'scale': [None, MaxAbsScaler(), StandardScaler(), MinMaxScaler()],
#         'reduce_dim': [SelectKBest()],
#         'reduce_dim__k': number_of_features,
#         'classify__metric': ["euclidean", "manhattan"],
#         'classify__shrink_threshold': shrink_threshold
#     }
# ]
# cv = StratifiedShuffleSplit(random_state=42)
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='precision', n_jobs=-1)
# features = f.df.as_matrix()[:, 0:-2]
# labels = f.df.as_matrix()[:, -1]
# grid.fit(features, labels)
# test_classifier(grid.best_estimator_, f.df.as_matrix())
# dump_classifier_and_data(grid.best_estimator_, f.df.as_matrix(), f.df.columns.values)
