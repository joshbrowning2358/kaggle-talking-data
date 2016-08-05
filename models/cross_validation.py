import pandas as pd
import numpy as np
import re
import csv

pd.options.mode.chained_assignment = None


class CrossValidation:
    """
    Class to perform automatic cross-validation when given data and a
    sampling approach.
    """

    def __init__(self, data, metric, y_col, train_col, cv_index_col=None, validation_index_col=None,
                 cv_time=False, id_col=None, logged=False):
        """
        :param data: A pandas object containing the training and testing data.
        :param metric: A function accepting two vectors and computing an error metric.
        :param cv_index_col: column name of train indicating which values should be used
        for training and which for validation (all are used for the final test).  If only two
        unique values are available, the first value will be assumed to correspond to the training
        set and the second the validation set (i.e. no cross-validation will be performed).
        Currently supported structures:
        {'type': shuffle, 'fold_cnt': numeric (optional)}
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be of type DataFrame!')
        if train_col not in data.columns:
            raise KeyError('train_col not in column names of data!')
        if y_col not in data.columns:
            raise KeyError('y_col not in column names of data!')
        self.train = data.ix[data[train_col].tolist(), :]
        self.test = data.ix[[not x for x in data[train_col].tolist()], :]
        self.y_col = y_col
        self.metric = metric
        if cv_index_col is not None and validation_index_col is None:
            self.fold_col = cv_index_col
            self.cv_type = 'cross_validation'
        elif cv_index_col is None and validation_index_col is not None:
            self.fold_col = validation_index_col
            self.cv_type = 'single_validation'
        elif cv_index_col is not None and validation_index_col is not None:
            raise ValueError('Cannot provide both cv_indices and validation_indices!')
        else:
            if 'fold_' in self.train.columns:
                raise ValueError('fold_ cannot be a column name of data!')
            self._assign_cv_fold()
            self.fold_col = 'fold_'
            self.cv_type = 'cross_validation'
        if self.fold_col not in self.train.columns:
            raise ValueError('fold column must be a valid column of data!')
        if cv_time:
            self.cv_type = 'time_validation'
        if id_col is not None:
            self.id_col = id_col
            if id_col not in self.test.columns:
                raise KeyError('id_col must be a column name of data!')
        else:
            if 'id_col' in self.test.columns:
                raise KeyError('"id_col" cannot be a column name of data!')
            self.test['id_col'] = range(self.test.shape[0])
            self.id_col = 'id_col'
        self.model_key = str(np.random.randint(100000)) + str(np.random.randint(100000))
        self.logged = logged

    def run(self, model, filename=None):
        """
        :param model: An instance with fit and predict methods.
        :param filename: Str, used to write out result files (from cv or train/test)
        """
        if filename is None:
            filename = "/data/models/" + self.model_key
        else:
            filename = re.sub("\\.csv", "_" + self.model_key, filename)
        if self.cv_type in ['time_validation', 'cross_validation']:
            score = self._run_cv(model, filename)
        elif self.cv_type == 'validation':
            score = self._run_validation(model, filename)
        else:
            raise TypeError('Validation type {} not yet implemented!'.format(self.cv_type))
        self._run_final(model, filename)
        if self.logged:
            print "Results of final prediction (fitting on full dataset) saved in " + filename
            self._log_results(filename, score)

    def _run_cv(self, model, filename):
        error_metric = pd.DataFrame({'fold': [], 'error': []})
        folds_to_run = self.train[self.fold_col].unique()
        cv_prediction = np.repeat(np.nan, len(self.train))
        for fold_number in folds_to_run:
            error = self._run_cv_fold(model, fold_number, cv_prediction)
            error_metric = error_metric.append({'fold': fold_number, 'error': error}, ignore_index=True)
            print 'Error for fold {} was: {}'.format(fold_number, str(error))
        total_error = round(self.metric(self.train.ix[:, self.y_col], cv_prediction), 6)
        if self.logged:
            file_ = re.sub('.csv', '', filename) + '_' + self.model_key + str(total_error) + '_cv.csv'
            out = pd.DataFrame({self.id_col: self.train[self.id_col], 'cv_prediction': cv_prediction})
            out.to_csv(file_, ',', header=True, cols=[self.id_col, 'cv_prediction'], index=False)
            print 'Cross validation results saved in ' + file_
        print 'Error was: ' + str(total_error)
        return error_metric

    def _run_validation(self, model, filename):
        fold_number = np.min(self.train[self.fold_col])
        cv_prediction = np.repeat(np.nan, len(self.train))
        error = self._run_cv_fold(model, fold_number, cv_prediction)
        error_metric = pd.DataFrame({'fold': [fold_number], 'error': [error]})
        if self.logged:
            file_ = re.sub('.csv', '', filename) + '_' + self.model_key + str(round(error, 6)) + '_cv.csv'
            out = pd.DataFrame({self.id_col: self.train[self.id_col], 'cv_prediction': cv_prediction})
            out.to_csv(file_, ',', header=True, cols=[self.id_col, 'cv_prediction'], index=False)
            print 'Validation results saved in ' + file_
        print 'Error was: ' + str(error)
        return error_metric

    def _run_final(self, model, filename):
        model.fit(X=self.train, y=self.train.ix[:, self.y_col])
        prediction = model.predict(self.test)
        if self.logged:
            file_ = re.sub('.csv', '', filename) + '_' + self.model_key + '_full.csv'
            out = pd.DataFrame({self.id_col: self.test[self.id_col], 'prediction': prediction})
            out.to_csv(file_, ',', header=True, cols=[self.id_col, 'prediction'], index=False)
            print 'Final predictions saved in ' + file_

    def _log_results(self, filename, score):
        print '\n'

    def _assign_cv_fold(self):
        fold_cnt = 4
        records_per_fold = self.train.shape[0] / fold_cnt + 1
        fold_numbers = np.random.permutation(range(fold_cnt)*records_per_fold)
        self.train['fold_'] = fold_numbers[:self.train.shape[0]]

    def _run_cv_fold(self, m, fold_number, cv_prediction):
        cv_filter = self.train[self.fold_col] == fold_number
        train_filter = [not x for x in cv_filter]
        m.fit(X=self.train.ix[train_filter, :], y=self.train.ix[train_filter, self.y_col])
        prediction = m.predict(X=self.train.ix[cv_filter, :])
        cv_prediction = [pred if cv else old for pred, old, cv in zip(prediction, cv_prediction, cv_filter)]
        return self.metric(self.train.ix[cv_filter, self.y_col], prediction)

    @staticmethod
    def write_csv(data, _file):
        data.to_csv(file, ',', header=True, engine='python')