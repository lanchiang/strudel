# this is the implementation of line classification version of strudel.
# Created by lan at 2020/3/2
import logging
from timeit import default_timer

import numpy as np
import pandas
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

logging.basicConfig(level=logging.INFO)


def create_feature_vector(dataset_name, feature_vector_path):
    """
    create the feature vector from the input data files.

    :param dataset_name: the name of datasets used for experiments.
    # :param data_path: the path that stores data files, each file is a single csv file.
    :return: feature vector with label for all data files, each line for a single csv file
    """
    if dataset_name == 'all':
        line_feature_vector_path = feature_vector_path + 'all/training.csv'
        line_metadata_path = feature_vector_path + 'all/metadata.csv'
    else:
        line_feature_vector_path = feature_vector_path + dataset_name + '/training.csv'
        line_metadata_path = feature_vector_path + dataset_name + '/metadata.csv'

    line_feature_vector_df = pandas.read_csv(line_feature_vector_path)
    line_metadata_df = pandas.read_csv(line_metadata_path)

    # create is derived line feature from the cell feature file.
    cell_feature_path = '../feature_vec/cstrudel/' + dataset_name + ".csv"
    cell_feature_df = pandas.read_csv(cell_feature_path)[['is_sum','filename', 'sheetname', 'row']]
    groups = cell_feature_df.groupby(['filename', 'sheetname', 'row'])
    is_sum_amount = []
    for index, group in groups:
        is_sum_amount.append([index[0], index[1], index[2], group[group['is_sum'] == 1].size / group.size])
    is_sum_amount_df = pandas.DataFrame(data=is_sum_amount, columns=['filename', 'sheetname', 'row', 'is_sum_ratio'])

    return line_feature_vector_df, line_metadata_df, is_sum_amount_df

    # return line_feature_vector_df, line_metadata_df, None


class LStrudel:

    algo_name = 'strudel_line'

    def __init__(self, data_path, dataset_name, feature_vec_path):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.feature_vec_path = feature_vec_path
        self.metadata = None
        self.result = pandas.DataFrame()
        self.runtime = 0

    def execute(self, method='predict_proba'):
        global clf, pred_proba_df
        logging.info('Running lstrudel...')
        logging.info('create feature vector...')
        line_feature_vector_df, line_metadata_df, is_sum_amount_df = create_feature_vector(self.dataset_name, self.feature_vec_path)
        self.metadata = line_metadata_df

        all_info_matrix = pandas.concat([line_feature_vector_df, line_metadata_df], axis=1)

        all_info_matrix = all_info_matrix[all_info_matrix['Label'] != 'empty']

        all_info_matrix = is_sum_amount_df.merge(all_info_matrix,
                                           left_on=['filename', 'sheetname', 'row'],
                                           right_on=['Excel file name', 'Spreadsheet name', 'Line number']
                                           ).drop(['filename', 'sheetname', 'row'], axis=1)

        print(np.unique(all_info_matrix['Label'], return_counts=True))

        print(all_info_matrix.columns)

        groups_by_file_sheet_name = all_info_matrix.groupby(['Excel file name', 'Spreadsheet name'])

        file_sheet_names = list(groups_by_file_sheet_name.groups.keys())

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        pred_labels = []
        line_metadata = []

        for train_index, test_index in skf.split(file_sheet_names, [0] * len(file_sheet_names)):
            train_cells = []
            test_cells = []

            train_set_file_sheet_names = [file_sheet_names[index] for index in train_index]
            for train_set_file_sheet_name in train_set_file_sheet_names:
                grp = groups_by_file_sheet_name.get_group(train_set_file_sheet_name)
                train_cells.extend(grp.to_records(index=False))

            test_set_file_sheet_names = [file_sheet_names[index] for index in test_index]
            for test_set_file_sheet_name in test_set_file_sheet_names:
                grp = groups_by_file_sheet_name.get_group(test_set_file_sheet_name)
                test_cells.extend(grp.to_records(index=False))

            train_df = pandas.DataFrame.from_records(train_cells, columns=all_info_matrix.columns)
            train_df = train_df.drop(self.metadata.columns, axis=1)

            test_df = pandas.DataFrame.from_records(test_cells, columns=all_info_matrix.columns)
            metadata = test_df[self.metadata.columns].to_records(index=False)
            line_metadata.extend(metadata)

            test_df = test_df.drop(self.metadata.columns, axis=1)

            X_train = train_df.iloc[:, 0: len(train_df.columns) - 1]
            y_train = train_df.iloc[:, len(train_df.columns) - 1:len(train_df.columns)]

            X_test = test_df.iloc[:, 0:len(test_df.columns) - 1]
            y_test = test_df.iloc[:, len(test_df.columns) - 1:len(test_df.columns)]

            clf = RandomForestClassifier(n_jobs=8)

            clf.fit(X_train, np.ravel(y_train))

            if method == 'predict_proba':
                pred = clf.predict_proba(X_test)
            else:
                pred = clf.predict(X_test)

            pred_labels.extend(pred)

        if method == 'predict_proba':
            pred_proba_df = pandas.DataFrame(data=pred_labels, columns=clf.classes_ + '_prob')
        else:
            pred_proba_df = pandas.DataFrame(data=pred_labels, columns=['pred_label'])

        line_metadata_df = pandas.DataFrame.from_records(data=line_metadata, columns=line_metadata_df.columns)

        return pred_proba_df, line_metadata_df


if __name__ == '__main__':
    data_path = './data/saus.jl.gz'
    dataset_name = 'saus'
    feature_vector_path = '/Users/lan/PycharmProjects/table-content-type-classification/feature_vec/lstrudel'
    feature_vector_path = feature_vector_path + '/' if not str(feature_vector_path).endswith('/') else (
            feature_vector_path)
    lstrudel_algorithm = LStrudel(data_path, dataset_name, feature_vector_path)
    pred_proba, metadata = lstrudel_algorithm.execute()
    pred_proba = pandas.concat([pred_proba, metadata], axis=1)
    pred_proba.to_csv('/Users/lan/PycharmProjects/table-content-type-classification/lstrudel_prob/' + dataset_name + '.csv', index=False)