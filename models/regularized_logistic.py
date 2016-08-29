# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from feature_engineering import FeatureEngineering
from cross_validation import CrossValidation
from multi_log_loss import multi_log_loss

f = FeatureEngineering('../data/gender_age_train.csv',
                       '../data/gender_age_test.csv',
                       'device_id',
                       wide_files=[#'../features/apps_per_event.csv', '../features/avg_position.csv',
                                   #'../features/count_by_hour.csv', '../features/count_by_period.csv',
                                   '../features/event_counts.csv', '../features/sd_position.csv'],
                       long_files=[#'../features/active_app_category_counts.csv',
                                   #'../features/installed_app_category_counts.csv',
                                   '../features/phone_brand.csv'])
labels, features, colnames = f.extract_features()
labels.set_index(np.arange(labels.shape[0]), inplace=True)
colnames.set_index(np.arange(colnames.shape[0]), inplace=True)
train_filter = [i for i, x in enumerate(labels['age'].tolist()) if not np.isnan(x)]
test_filter = [i for i, x in enumerate(labels['age'].tolist()) if np.isnan(x)]

cv = CrossValidation(features[train_filter, :],
                     labels.ix[train_filter, 'group'],
                     features[test_filter, :],
                     multi_log_loss)
model = MultinomialNB()
model.predict = model.predict_proba
out = cv.run(model, 'test')