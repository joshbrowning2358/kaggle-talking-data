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
        self._get_train_test()
        if self.wide_files is not None:
            self._add_wide_features()
        if self.long_files is not None:
            self._add_long_features()
        return self.data

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
            row_indices = new_features[self.key]
            row_indices.drop_duplicates(inplace=True)
            row_indices['row_index'] = range(row_indices.shape[0])
            col_indices = set(new_features['variable'].tolist())
            col_indices = pd.DataFrame({'variable': list(col_indices), 'col_index': range(len(col_indices))})
            new_features = new_features.merge(row_indices, on=self.key)
            new_features = new_features.merge(col_indices, on='variable')
            sparse_mat = csr_matrix((new_features['value'], (new_features['row_index'], new_features['col_index'])),
                                    shape=(row_indices.shape[0], col_indices.shape[0]))
            new_features = pd.SparseDataFrame(
                [pd.SparseSeries(sparse_mat[i].toarray().ravel())
                 for i in np.arange(sparse_mat.shape[0])])
            new_features.columns = col_indices['variable'].tolist()
            # The indices will line up since row_indices is sorted by row_index (by construction) and these row_indices
            # control the definition of sparse_mat.
            row_indices.index = new_features.index
            new_features = pd.concat([new_features, row_indices[self.key]], axis=1)
            self.data = self.data.merge(new_features, 'left', self.key)

    def _add_wide_features(self):
        for f in self.wide_files:
            new_features = pd.read_csv(f)
            self._check_wide_feature(new_features, f)
            self.data = self.data.merge(new_features, 'left', self.key)

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
