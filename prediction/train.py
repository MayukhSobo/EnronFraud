from feature_engineering import feature_loader
from feature_engineering import feature_misc

# ------------ Creating a Feature Object ------------ #
f = feature_loader.FeatureExtract()
# ------------ Creating a Axillary  feature operation Object ---------- #
aux = feature_misc.Aux('create', f)
#  ------------ Create a new feature ------------ #
aux.operate('+', 'new_feature', 'bonus', 'salary')
#  ------------ Split the dataset for train and test --------- #
f.feature_splits()

