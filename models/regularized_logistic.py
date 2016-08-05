from sklearn.linear_model import LogisticRegression

from feature_engineering import FeatureEngineering
from cross_validation import CrossValidation

f = FeatureEngineering('../data/gender_age_train.csv',
                       '../data/gender_age_test.csv',
                       'device_id',
                       long_files=['../features/phone_brand.csv'])
d = f.extract_features()