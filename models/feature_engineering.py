import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

pd.options.mode.chained_assignment = None

class FeatureEngineering():

    def __init__(self, train_file, test_file, key, wide_files=None, long_files=None):
        """

        :param train_file: A string with the file containing training data
        :param test_file: A string with the file containing testing data
        :param key: A list of column names of train and test which represents a unique observation (used for joins)
        :param long_files: A string or list of strings with the file(s) containing the long format features
        :param wide_files: A string or list of strings with the file(s) containing the wide format features.  Note that
        any missing keys will be assumed to be 0.
        """
        self.train_file = train_file
        self.test_file = test_file
        if isinstance(long_files, str):
            long_files = [long_files]
        self.long_files = long_files
        if isinstance(wide_files, str):
            wide_files = [wide_files]
        self.wide_files = wide_files
        self._check_inputs_are_strings()
        self.data = None
        self.features = None
        if isinstance(key, str):
            key = [key]
        self.key = key

    def _check_inputs_are_strings(self):
        if not isinstance(self.train_file, str):
            raise TypeError('train_file must be a string!')
        if not isinstance(self.test_file, str):
            raise TypeError('train_file must be a string!')
        if self.long_files is not None:
            for i in range(len(self.long_files)):
                if not isinstance(self.long_files[i], str):
                    raise TypeError('long_files must be a string or list of strings, but element' +
                                    ' {} is of type {}!'.format(i, type(self.long_files[i])))
        if self.wide_files is not None:
            for i in range(len(self.wide_files)):
                if not isinstance(self.wide_files[i], str):
                    raise TypeError('wide_files must be a string or list of strings, but element' +
                                    ' {} is of type {}!'.format(i, type(self.wide_files[i])))

    def extract_features(self):
        """
        Extracts the features from the provided files.
        :return: A tuple of:
            - self.data: The train and test datasets bound together, retaining the original features from those files.
            This is intended for providing the target.
            - features: A sparse matrix (of class csr_matrix) that contains the features.  The rows correspond to
            the same rows in self.data, and the column names are provided in col_indices.
            - col_indices: The indices of the columns in the features object
        """
        self._get_train_test()
        if self.wide_files is not None:
            self._add_wide_features()
        if self.long_files is not None:
            self._add_long_features()
        col_indices, features = self._cast_long_features_to_wide()
        return self.data, features, col_indices

    def _get_train_test(self):
        train = pd.read_csv(self.train_file)
        test = pd.read_csv(self.test_file)
        if set(self.key).intersection(train) != set(self.key):
            raise TypeError('Not all keys were found in column names of train!')
        if set(self.key).intersection(test) != set(self.key):
            raise TypeError('Not all keys were found in column names of test!')
        self.data = pd.concat([train, test])

    def _add_long_features(self):
        for f in self.long_files:
            new_features = pd.read_csv(f)
            self._check_long_feature(new_features, f)
            self.features = pd.concat([self.features, new_features], axis=0)

    def _add_wide_features(self):
        for f in self.wide_files:
            new_features = pd.read_csv(f, delimiter=',')
            self._check_wide_feature(new_features, f)
            non_key_cols = list(set(new_features.columns.tolist()).difference(self.key))
            for col in non_key_cols:
                data_to_add = new_features[self.key + [col]]
                data_to_add['variable'] = col
                data_to_add.rename(columns={col: 'value'}, inplace=True)
                self.features = pd.concat([self.features, data_to_add], axis=0)

    def _check_wide_feature(self, feature, filename):
        if set(self.key).intersection(feature) != set(self.key):
            raise TypeError('Not all keys were found in column names of {}!'.format(filename))
        if set(feature.columns).intersection(self.data.columns) != set(self.key):
            raise TypeError('Already defined variables are in feature {}!'.format(filename))

    def _check_long_feature(self, feature, filename):
        if set(self.key).intersection(feature.columns) != set(self.key):
            raise TypeError('Not all keys were found in column names of {}!'.format(filename))
        if {'variable', 'value'}.intersection(feature.columns) != {'variable', 'value'}:
            raise TypeError("All long features must have columns 'variable' and 'value'!  Check {}".format(filename))
        if len(feature.columns) > len(self.key) + len({'variable', 'value'}):
            raise TypeError("Unexpected columns (not keys or 'variable'/'value') in {}".format(filename))
        overlap = set(feature['variable']).intersection(self.data.columns)
        if overlap:
            raise TypeError('Re-defining variables {} in feature file {}!'.format(overlap, filename))

    def _cast_long_features_to_wide(self):
        row_indices = self.data[self.key]
        if self.data[self.key].duplicated().any():
            raise KeyError('Multiple occurences of one key were found in train/test!  The key must be unique!')
        row_indices['row_index'] = range(row_indices.shape[0])
        col_indices = set(self.features['variable'].tolist())
        col_indices = pd.DataFrame({'variable': list(col_indices), 'col_index': range(len(col_indices))})
        self.features = self.features.merge(row_indices, on=self.key)
        self.features = self.features.merge(col_indices, on='variable')
        sparse_mat = csr_matrix((self.features['value'], (self.features['row_index'], self.features['col_index'])),
                                shape=(row_indices.shape[0], col_indices.shape[0]))
        # The indices will line up since row_indices is sorted by row_index (by construction) and these row_indices
        # control the definition of sparse_mat.
        return col_indices, sparse_mat
