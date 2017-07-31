from feature_engineering import feature_loader
from feature_engineering import feature_misc
from feature_engineering import feature_importance


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
print imp.get_importance_rf(save=False).keys()
print imp.get_importance_xgboost(save=False).keys()
print imp.get_importance_kBest(k=5, eval_func='classif').keys()
