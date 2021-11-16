# this is the implementation of line classification version of strudel.
# Created by lan at 2020/3/2
# import logging
import string
from functools import reduce

import numpy as np
import pandas
import rolling
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ndcg_score
from sklearn.model_selection import StratifiedKFold

from src.utility import is_numeric, detect_datatype, calculate_hist_diff


def is_empty_row(row):
    return all([True if cell.strip() == '' else False for cell in row])


def create_line_feature_vector(file_json_dict):
    file_array = np.array(file_json_dict['table_array'])

    aggregation_keywords = ['sum', 'total', 'average', 'avg']

    num_lookup_lines = 5

    def has_aggregation_keyword(row):
        tokens = map(lambda w: w.lower(), reduce(list.__add__, [word_tokenize(cell) for cell in row]))
        kw_exist = 1 if len(set(tokens).intersection(aggregation_keywords)) > 0 else 0
        return kw_exist

    f_kw_exist = np.apply_along_axis(func1d=has_aggregation_keyword, axis=1, arr=file_array)

    def row_emptiness_pattern(row):
        pattern = np.array([1 if len(cell.strip()) != 0 else 0 for cell in row])
        return pattern

    def discounted_cumulative_gain(row):
        # We hope all "1" are at the left of a pattern to all "0". Translating it to a dcg calculation,
        # an element's relevance is always 1 more than the element to its right, and the right-most element has a relevance score of 1.
        # For example, given a pattern [1,1,0,1,0,0], the relevance of the sequence is [6,5,4,3,2,1]
        # For an ideal pattern, the relevance of all "1" is large than that of all "0".
        re_pattern = np.asarray([row_emptiness_pattern(row)])
        pattern_len = re_pattern.shape[1]
        relevance = np.asarray([[pattern_len - index for index in range(0, pattern_len)]])
        n_dcg = ndcg_score(re_pattern, relevance)
        return n_dcg

    f_dcg = np.apply_along_axis(func1d=discounted_cumulative_gain, axis=1, arr=file_array)

    def empty_cell_ratio(row):
        re_pattern = row_emptiness_pattern(row)
        empty_cell_ratio = 1 - np.count_nonzero(re_pattern) / len(row)
        return empty_cell_ratio

    f_empty_cell_ratio = np.apply_along_axis(func1d=empty_cell_ratio, axis=1, arr=file_array)

    def numeric_cell_ratio(row):
        nc_ratio = len([cell for cell in row if is_numeric(cell)]) / len(row)
        return nc_ratio

    f_numeric_cell_ratio = np.apply_along_axis(func1d=numeric_cell_ratio, axis=1, arr=file_array)

    def string_cell_ratio(row):
        sc_ratio = len([cell for cell in row if detect_datatype(cell) == 4]) / len(row)
        return sc_ratio

    f_string_cell_ratio = np.apply_along_axis(func1d=string_cell_ratio, axis=1, arr=file_array)

    def average_word_count(row):
        cell_tokens = [list(filter(lambda token: token not in string.punctuation, word_tokenize(cell))) for cell in row]
        len_cell_tokens = [len(tokens) for tokens in cell_tokens if len(tokens) > 0]
        if not len_cell_tokens:
            awc = 0.0
        else:
            awc = np.sum(len_cell_tokens) / len(len_cell_tokens)
        return awc

    f_average_word_count = np.apply_along_axis(func1d=average_word_count, axis=1, arr=file_array)

    def empty_line_ratio_after(window):
        row_window = window[0]
        rows_after = row_window[1:]
        empty_rows_after = [row_after for row_after in rows_after if is_empty_row(row_after)]
        if not rows_after:
            ratio = 0.0
        else:
            ratio = len(empty_rows_after) / len(rows_after)
        return ratio

    windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
    windows = np.array([windows[len(windows) - len(file_array):]], dtype=object).transpose()

    f_empty_line_ratio_after = np.apply_along_axis(func1d=empty_line_ratio_after, axis=1, arr=windows)

    def empty_line_ratio_before(window):
        row_window = window[0]
        rows_before = row_window[:-1]
        empty_row_before = [row_before for row_before in rows_before if is_empty_row(row_before)]
        if not rows_before:
            ratio = 0.0
        else:
            ratio = len(empty_row_before) / len(rows_before)
        return ratio

    windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
    windows = np.array([windows[:len(file_array)]], dtype=object).transpose()

    f_empty_line_ratio_before = np.apply_along_axis(func1d=empty_line_ratio_before, axis=1, arr=windows)

    def hist_diff_after(window):
        row_window = window[0]
        row = row_window[0]
        row_value_length = [len(cell.strip()) for cell in row]
        rows_after = row_window[1:]
        hist_diff = 0.0
        for row_after in rows_after:
            if is_empty_row(row_after):
                continue
            row_after_value_length = [len(cell.strip()) for cell in row_after]
            hist_diff = calculate_hist_diff(row_value_length, row_after_value_length)
            break
        return hist_diff

    windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
    windows = np.array([windows[-len(file_array):]], dtype=object).transpose()

    f_hist_diff_after = np.apply_along_axis(func1d=hist_diff_after, axis=1, arr=windows)

    def hist_diff_before(window):
        row_window = window[0]
        row = row_window[-1]
        row_value_length = [len(cell.strip()) for cell in row]
        rows_before = row_window[:-1]
        hist_diff = 0.0
        for row_before in reversed(rows_before):
            if is_empty_row(row_before):
                continue
            row_before_value_length = [len(cell.strip()) for cell in row_before]
            hist_diff = calculate_hist_diff(row_value_length, row_before_value_length)
            break
        return hist_diff

    windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
    windows = np.array([windows[:len(file_array)]], dtype=object).transpose()

    f_hist_diff_before = np.apply_along_axis(func1d=hist_diff_before, axis=1, arr=windows)

    def data_type_after(window):
        row_window = window[0]
        row = row_window[0]
        row_data_type = [detect_datatype(cell) for cell in row]
        rows_after = row_window[1:]
        data_type_same_ratio = 0.0
        for row_after in rows_after:
            if is_empty_row(row_after):
                continue
            row_after_data_type = [detect_datatype(cell) for cell in row_after]
            cell_same_data_type = [(this_cell, that_cell) for this_cell, that_cell in zip(row_data_type, row_after_data_type) if this_cell == that_cell]
            data_type_same_ratio = len(cell_same_data_type) / len(row_data_type)
        return data_type_same_ratio

    windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
    windows = np.array([windows[-len(file_array):]], dtype=object).transpose()

    f_data_type_after = np.apply_along_axis(func1d=data_type_after, axis=1, arr=windows)

    def data_type_before(window):
        row_window = window[0]
        row = row_window[-1]
        row_data_type = [detect_datatype(cell) for cell in row]
        rows_before = row_window[:-1]
        data_type_same_ratio = 0.0
        for row_before in rows_before:
            if is_empty_row(row_before):
                continue
            row_before_data_type = [detect_datatype(cell) for cell in row_before]
            cell_same_data_type = [(this_cell, that_cell) for this_cell, that_cell in zip(row_data_type, row_before_data_type) if this_cell == that_cell]
            data_type_same_ratio = len(cell_same_data_type) / len(row_data_type)
        return data_type_same_ratio

    windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
    windows = np.array([windows[:len(file_array)]], dtype=object).transpose()

    f_data_type_before = np.apply_along_axis(func1d=data_type_before, axis=1, arr=windows)



    lstrudel_feature_vector = pandas.DataFrame({'has_derived_keyword': f_kw_exist,
                                                'discounted_cumulative_gain': f_dcg,
                                                'line_empty_cell_ratio': f_empty_cell_ratio,
                                                'line_numeric_cell_ratio': f_numeric_cell_ratio,
                                                'line_string_cell_ratio': f_string_cell_ratio,
                                                'line_average_word_count': f_average_word_count,
                                                'empty_line_ratio_after': f_empty_line_ratio_after,
                                                'empty_line_ratio_before': f_empty_line_ratio_before,
                                                'hist_diff_after': f_hist_diff_after,
                                                'hist_diff_before': f_hist_diff_before,
                                                'data_type_after': f_data_type_after,
                                                'data_type_before': f_data_type_before})

    return lstrudel_feature_vector


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
    cell_feature_df = pandas.read_csv(cell_feature_path)[['is_sum', 'filename', 'sheetname', 'row']]
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
        # logging.info('Running lstrudel...')
        # logging.info('create feature vector...')
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
    data_path = '../data/saus.jl.gz'
    dataset_name = 'saus'
    feature_vector_path = '/Users/lan/PycharmProjects/table-content-type-classification/feature_vec/lstrudel'
    feature_vector_path = feature_vector_path + '/' if not str(feature_vector_path).endswith('/') else (
        feature_vector_path)
    lstrudel_algorithm = LStrudel(data_path, dataset_name, feature_vector_path)
    pred_proba, metadata = lstrudel_algorithm.execute()
    pred_proba = pandas.concat([pred_proba, metadata], axis=1)
    pred_proba.to_csv('/Users/lan/PycharmProjects/table-content-type-classification/lstrudel_prob/' + dataset_name + '.csv', index=False)
