from sklearn.linear_model import LogisticRegression
import numpy as np

from feature_engineering import FeatureEngineering
from cross_validation import CrossValidation

f = FeatureEngineering('../data/gender_age_train.csv',
                       '../data/gender_age_test.csv',
                       'device_id',
                       wide_files=['../features/apps_per_event.csv', '../features/avg_position.csv',
                                   '../features/count_by_hour.csv', '../features/count_by_period.csv',
                                   '../features/event_counts.csv', '../features/sd_position.csv'],
                       long_files=['../features/active_app_category_counts.csv',
                                   '../features/installed_app_category_counts.csv',
                                   '../features/phone_brand.csv'])
d = f.extract_features()
data = d[0]['group']
train_filter = [i for i, x in enumerate(d[0]['age'].tolist()) if not np.isnan(x)]
test_filter = [i for i, x in enumerate(d[0]['age'].tolist()) if np.isnan(x)]

cv = CrossValidation(d[1][train_filter, :], d[1][test_filter, :], metric)