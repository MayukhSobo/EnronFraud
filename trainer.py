from feature_engineering import feature_loader
from feature_engineering import feature_misc
from feature_engineering import feature_importance
from sklearn.pipeline import Pipeline


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
aux.operate('/', 'from_this_person_to_poi_stand', 'from_this_person_to_poi', 'from_messages')
aux.operate('/', 'from_poi_to_this_person_stand', 'from_poi_to_this_person', 'to_messages')
aux.operate('+', 'poi_interaction', 'from_poi_to_this_person_stand', 'from_this_person_to_poi_stand')

'''
Feature Name        : income_ratio
Feature Expression  : (salary + bonus + long_term_incentive) / total_payment

Description         : This is very powerful feature because it standardise the income of an employee
                      with the ```salary``` and ```bonus```. The point is that is the person is not a
                      POI then, this ratio wouldn't be too large but if a person is POI, this value
                      would probably be quite large.
'''
aux.operate('+', 'bonusPlusSalaryPlusIncentives', 'bonus', 'salary', 'long_term_incentive')
aux.operate('/', 'income_ratio', 'bonusPlusSalaryPlusIncentives', 'total_payments')

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
           'from_poi_to_this_person_stand'],
          axis=1, inplace=True)
#  ------------ Split the dataset for train and test --------- #
f.feature_splits()
# -------------- Feature Selection ---------------- #
#
imp = feature_importance.Importance(algo='*', fObj=f)
# print imp.get_importance_rf(save=False).keys()
# print imp.get_importance_xgboost(save=False).keys()
# print imp.get_importance_kBest(k=5, eval_func='classif').keys()

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
# from sklearn.naive_bayes import GaussianNB
#
features = imp.get_importance_rf(save=False).keys()
X_train, X_test = f.adhoc_feature_parse(columns=features)






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

# Code would go here

# Model 3

'''
    Feature Selection:
        algo: PCA
        type: auto
        n_components: 3
        whiten: True

    Feature Scaling:
        MaxAbsScaler

    Cross Validation:
        StratifiedShuffleSplit

    Classification:
        algo: KNeighborsClassifier
'''

# Code would go here

# Model 4

'''
    Feature Selection:
        algo: K Best
        k: 5
        eval_func: f_classif

    Feature Scaling:
        MinMaxScaler

    Cross Validation:
        None

    Classification:
        algo: SVC
'''

# Code would go here

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
        MinMaxScaler

    Cross Validation:
        StratifiedShuffleSplit

    Classification:
        algo: Gradient Boosting
'''

# Code would go here

