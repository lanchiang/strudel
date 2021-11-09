# 
# Created by lan at 2020/3/3
from datetime import timedelta
from timeit import default_timer

import numpy as np
import pandas
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
# import cstrudel_features
# from data import load_data
from tqdm import tqdm

from block_size_calculator import cal_block_size_linewise
from derived_detector import cal_is_derived
from lstrudel import LStrudel
from utility import detect_datatype


class cstrudel:
    algorithm = 'strudel_line'

    def __init__(self, data_folder, dataset_name, feature_vector_path, cstrudel_component='all'):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.feature_vector_path = feature_vector_path
        self.cstrudel_component = cstrudel_component
        self.runtime = 0
        self.result = pandas.DataFrame()
        pass

    def train(self):
        lstrudel_fvec_path = self.feature_vector_path + 'lstrudel/' if str(self.feature_vector_path).endswith('/') else (
                self.feature_vector_path + '/lstrudel/')

        # result of line classification as prediction probabilities
        lstrudel_algorithm = LStrudel(self.data_folder, self.dataset_name, lstrudel_fvec_path)
        pred_proba, metadata = lstrudel_algorithm.execute()
        pred_proba = pandas.concat([pred_proba, metadata], axis=1)

        cstrudel_fvec_path = self.feature_vector_path + 'cstrudel/' if str(self.feature_vector_path).endswith('/') else (
                self.feature_vector_path + '/cstrudel/')
        cstrudel_fvec_path = cstrudel_fvec_path + self.dataset_name + '.csv'

        cell_features_df = pandas.read_csv(cstrudel_fvec_path, header=0)
        print(cell_features_df.columns)

        cell_labels_df = cell_features_df.iloc[:, len(cell_features_df.columns) - 1: len(cell_features_df.columns)]
        cell_features_df = cell_features_df.iloc[:, 0:len(cell_features_df.columns) - 1]

        # combine cell features with line type prediction probabilities
        cell_features_df = cell_features_df.merge(pred_proba,
                                                  left_on=['filename', 'sheetname', 'row'],
                                                  right_on=['Excel file name', 'Spreadsheet name', 'Line number'])
        cell_features_df['Line type'] = cell_labels_df

        all_info_matrix = cell_features_df.drop(columns=['Excel file name', 'Spreadsheet name', 'Line number',
                                                         'filename', 'sheetname', 'row', 'column'])

        X_train = all_info_matrix.iloc[:, 0: len(all_info_matrix.columns) - 1]
        y_train = all_info_matrix.iloc[:, len(all_info_matrix.columns) - 1:len(all_info_matrix.columns)]

        clf = RandomForestClassifier(n_jobs=8)

        pipeline = Pipeline([
            # ('under-sampling', usampler),
            ('classifying', clf)
        ], verbose=True)
        pipeline.fit(X_train, np.ravel(y_train))

        classes = clf.classes_

        feature_importance_list = []

        for single_class in classes:
            copied_infor_matrix = all_info_matrix.copy()

            copied_infor_matrix.loc[copied_infor_matrix['Line type'] != single_class, 'Line type'] = 'others'

            print(np.unique(copied_infor_matrix['Line type'], return_counts=True))

            X_train = copied_infor_matrix.iloc[:, 0: len(copied_infor_matrix.columns) - 1]
            y_train = copied_infor_matrix.iloc[:, len(copied_infor_matrix.columns) - 1:len(copied_infor_matrix.columns)]

            clf = RandomForestClassifier(n_jobs=8)

            clf.fit(X_train, np.ravel(y_train))

            perm_importance_result = permutation_importance(clf, X_train, y_train, n_repeats=5, random_state=0, n_jobs=8)

            print(perm_importance_result.importances_mean)

            print(clf.feature_importances_)

            feature_importance_list.append(perm_importance_result.importances_mean)

        pandas.DataFrame(data=feature_importance_list, index=classes, columns=all_info_matrix.drop(columns=['Line type']).columns).to_csv(
            './cell_feature_importance.csv')

        # from joblib import dump
        # dump(pipeline, 'cstrudel.model')

        return pipeline

    def fit_file(self, file):
        pass

    def run(self):
        start = default_timer()

        lstrudel_fvec_path = self.feature_vector_path + 'lstrudel/' if str(self.feature_vector_path).endswith('/') else (
                    self.feature_vector_path + '/lstrudel/')

        # result of line classification as prediction probabilities
        lstrudel_algorithm = LStrudel(self.data_folder, self.dataset_name, lstrudel_fvec_path)
        pred_proba, metadata = lstrudel_algorithm.execute()
        pred_proba = pandas.concat([pred_proba, metadata], axis=1)

        cstrudel_fvec_path = self.feature_vector_path + 'cstrudel/' if str(self.feature_vector_path).endswith('/') else (
                    self.feature_vector_path + '/cstrudel/')
        cstrudel_fvec_path = cstrudel_fvec_path + self.dataset_name + '.csv'

        cell_features_df = pandas.read_csv(cstrudel_fvec_path, header=0)
        print(cell_features_df.columns)

        cell_labels_df = cell_features_df.iloc[:, len(cell_features_df.columns) - 1: len(cell_features_df.columns)]
        cell_features_df = cell_features_df.iloc[:, 0:len(cell_features_df.columns) - 1]

        # combine cell features with line type prediction probabilities
        cell_features_df = cell_features_df.merge(pred_proba,
                                                  left_on=['filename', 'sheetname', 'row'],
                                                  right_on=['Excel file name', 'Spreadsheet name', 'Line number'])
        cell_features_df['Line type'] = cell_labels_df

        all_info_matrix = cell_features_df.drop(columns=['Excel file name', 'Spreadsheet name', 'Line number'])

        # is_sum_mark = all_info_matrix['is_sum']

        # is_sum_mark = is_sum_mark[is_sum_mark == 1].index

        # if 'is_sum' in all_info_matrix.columns:
        #     all_info_matrix = all_info_matrix.drop(columns=['is_sum'])

        groups_by_file_sheet_name = all_info_matrix.groupby(['filename', 'sheetname'])

        file_sheet_names = list(groups_by_file_sheet_name.groups.keys())

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        true_labels = []
        pred_labels = []

        result = []

        result_df_list = []

        iteration = 0
        # split data into training and test sets on the file level.
        for train_index, test_index in skf.split(file_sheet_names, [0] * len(file_sheet_names)):
            print("Iteration: " + str(iteration))
            iteration += 1
            train_cells = []
            test_cells = []

            train_set_file_sheet_names = [file_sheet_names[index] for index in train_index]
            train_cell_groups = []
            for train_set_file_sheet_name in train_set_file_sheet_names:
                grp = groups_by_file_sheet_name.get_group(train_set_file_sheet_name)
                train_cell_groups.append(grp)
                train_cells.extend(grp.to_records(index=False))

            train_cell_df = pandas.concat(train_cell_groups)
            train_cell_df = train_cell_df.drop(['filename', 'sheetname', 'row', 'column'], axis=1)

            test_set_file_sheet_names = [file_sheet_names[index] for index in test_index]
            test_cell_groups = []
            for test_set_file_sheet_name in test_set_file_sheet_names:
                grp = groups_by_file_sheet_name.get_group(test_set_file_sheet_name)
                test_cell_groups.append(grp)
                test_cells.extend(grp.to_records(index=False))

            test_cell_df = pandas.concat(test_cell_groups)
            test_metadata_df = test_cell_df[['filename', 'sheetname', 'row', 'column']]
            test_cell_df = test_cell_df.drop(['filename', 'sheetname', 'row', 'column'], axis=1)

            train_df = pandas.DataFrame.from_records(train_cells, columns=all_info_matrix.columns)
            # train_df = train_df.drop(['filename', 'sheetname', 'row', 'column'], axis=1)

            # append the instances in test set into the result list
            result.extend(test_cells)

            test_df = pandas.DataFrame.from_records(test_cells, columns=all_info_matrix.columns)
            # test_df = test_df.drop(['filename', 'sheetname', 'row', 'column'], axis=1)

            X_train_cell = train_cell_df.iloc[:, 0: len(train_cell_df.columns) - 1]
            y_train_cell = train_cell_df.iloc[:, len(train_cell_df.columns) - 1:len(train_cell_df.columns)]

            X_train = train_df.iloc[:, 0: len(train_df.columns) - 1]
            y_train = train_df.iloc[:, len(train_df.columns) - 1:len(train_df.columns)]

            X_test_cell = test_cell_df.iloc[:, 0: len(test_cell_df.columns) - 1]
            y_test_cell = test_cell_df.iloc[:, len(test_cell_df.columns) - 1:len(test_cell_df.columns)]

            X_test = test_df.iloc[:, 0:len(test_df.columns) - 1]
            y_test = test_df.iloc[:, len(test_df.columns) - 1:len(test_df.columns)]

            # train a random forest classifier for cell classification
            clf = RandomForestClassifier(n_jobs=8)

            pipeline = Pipeline([
                # ('under-sampling', usampler),
                ('classifying', clf)
            ], verbose=True)
            # pipeline.fit(X_train, np.ravel(y_train))
            pipeline.fit(X_train_cell, np.ravel(y_train_cell))

            # clf.fit(X_train, np.ravel(y_train))

            # predict the test set with the trained model.
            # pred = clf.predict(X_test)
            # pred = pipeline.predict(X_test)
            pred = pipeline.predict(X_test_cell)

            pred_df = pandas.DataFrame(data=pred, columns=['pred_label'], index=y_test_cell.index)

            true_pred_df = y_test_cell.merge(pred_df, left_index=True, right_index=True)

            true_pred_df = pandas.concat([test_metadata_df, true_pred_df], axis=1)

            result_df_list.append(true_pred_df)

            # test set ground truth
            y_test = [value[0] for value in y_test.values]

            # add the ground truth to the global ground truth list
            true_labels.extend(y_test)

            # add the prediction to the global prediction list
            pred_labels.extend(pred)

        result_df = pandas.concat(result_df_list)

        result_df.sort_index(inplace=True)

        # result_df.loc[is_sum_mark, 'pred_label'] = 'derived'

        true_labels = result_df['Line type']
        pred_labels = result_df['pred_label']

        # self.result = pandas.DataFrame.from_records(result, columns=all_info_matrix.columns)
        # self.result['pred_label'] = pred_labels
        self.result = result_df

        end = default_timer()
        self.runtime = end - start

        return true_labels, pred_labels


def create_features(_all_table_cells, _all_table_annotations, _all_filenames, _all_sheetnames):
    _flattened_feature_vectors = []
    _flattened_annotations = []
    _flattened_metadata = []
    _feature_names = ['row_position', 'column_position', 'value_length', 'data_type',
                      'empty_row_before', 'empty_row_after', 'empty_column_left', 'empty_column_right',
                      'derived_kw_line', 'derived_kw_column', 'empty_cell_ratio_row', 'empty_cell_ratio_column',
                      'normalized_block_size', 'is_sum',
                      'value_length_pos1', 'data_type_pos1', 'value_length_pos2', 'data_type_pos2',
                      'value_length_pos3', 'data_type_pos3', 'value_length_pos4', 'data_type_pos4',
                      'value_length_pos5', 'data_type_pos5', 'value_length_pos6', 'data_type_pos6',
                      'value_length_pos7', 'data_type_pos7', 'value_length_pos8', 'data_type_pos8'
                      ]

    derived_keywords = ['total', 'percentage', 'all', 'average', 'sum']

    runtime_create_features = []

    for i in range(1):
        _flattened_feature_vectors = []
        _flattened_annotations = []
        _flattened_metadata = []

        runtime_derived_cell_detection = 0

        avg_num_derived_candidates = 0

        for (table_cells, table_annotations, filename, sheetname) in tqdm(
                zip(_all_table_cells, _all_table_annotations, _all_filenames, _all_sheetnames)):
            # if filename != 'C10071':
            #     continue

            identifier_file = str(filename) + '@' + str(sheetname)

            # print('Start creating features for file: ' + identifier_file + '. File size: ' + str(len(table_cells)) + ' * ' + str(len(table_cells[0])))
            start = default_timer()
            # print(start)

            block_size = cal_block_size_linewise(table_annotations)

            table_length = len(table_cells)
            table_width = 0 if table_length == 0 else len(table_cells[0])
            table_size = table_length * table_width

            max_value_length = max([len(str.strip(cell)) for table_line in table_cells for cell in table_line])

            table_data_types = []
            for line in table_cells:
                line_data_types = [detect_datatype(cell) for cell in line]
                table_data_types.append(line_data_types)

            table_value_length = []
            for line in table_cells:
                line_value_length = [len(str.strip(cell)) for cell in line]
                table_value_length.append(line_value_length)

            num_numeric_by_row_index = {}
            for row_index in range(table_length):
                row_data_type = table_data_types[row_index]
                num_numeric_by_row_index[row_index] = len([cell_dt for cell_dt in row_data_type if cell_dt == 1 or cell_dt == 2])

            num_numeric_by_column_index = {}
            for column_index in range(table_width):
                column_data_type = [line[column_index] for line in table_data_types]
                num_numeric_by_column_index[column_index] = len([cell_dt for cell_dt in column_data_type if cell_dt == 1 or cell_dt == 2])

            row_emptiness_dict = {}
            column_emptiness_dict = {}

            derived_keyword_line_exist_dict = {}
            derived_keyword_column_exist_dict = {}

            start = default_timer()
            is_sum, num_derived_candidates = cal_is_derived(table_cells, aggr_delta=1.0E-1)
            avg_num_derived_candidates += num_derived_candidates
            end = default_timer()
            print('Calculate the is sum feature: ' + str(timedelta(seconds=end - start)))
            runtime_derived_cell_detection += end - start

            print('Calculate features for cells...')
            for row_index in range(table_length):
                for column_index in range(table_width):

                    if table_annotations[row_index][column_index] == 'empty':
                        continue

                    feature_vector = []

                    # normalized row position
                    row_position_feature = row_index / table_length
                    feature_vector.append(row_position_feature)

                    # normalized column position
                    column_position_feature = column_index / table_width
                    feature_vector.append(column_position_feature)

                    # value length
                    value_length_feature = len(str.strip(table_cells[row_index][column_index])) / max_value_length
                    feature_vector.append(value_length_feature)

                    # data type
                    value_data_type_feature = detect_datatype(table_cells[row_index][column_index])
                    feature_vector.append(value_data_type_feature)

                    # is empty line above
                    above_line_empty_feature = 0
                    if row_index != 0:
                        if row_index - 1 in row_emptiness_dict:
                            non_empty_cells = row_emptiness_dict[row_index - 1]
                        else:
                            non_empty_cells = [cell for cell in table_annotations[row_index - 1] if cell != 'empty']
                            row_emptiness_dict[row_index - 1] = non_empty_cells
                        if len(non_empty_cells) == 0:
                            above_line_empty_feature = 1
                    feature_vector.append(above_line_empty_feature)

                    # is empty line after
                    after_line_empty_feature = 0
                    if row_index != table_length - 1:
                        if row_index + 1 in row_emptiness_dict:
                            non_empty_cells = row_emptiness_dict[row_index + 1]
                        else:
                            non_empty_cells = [cell for cell in table_annotations[row_index + 1] if cell != 'empty']
                            row_emptiness_dict[row_index + 1] = non_empty_cells
                        if len(non_empty_cells) == 0:
                            after_line_empty_feature = 1
                    feature_vector.append(after_line_empty_feature)

                    # empty column left
                    left_column_empty_feature = 0
                    if column_index != 0:
                        if column_index - 1 in column_emptiness_dict:
                            non_empty_cells = column_emptiness_dict[column_index - 1]
                        else:
                            left_column = [line[column_index - 1] for line in table_annotations]
                            non_empty_cells = [cell for cell in left_column if cell != 'empty']
                            column_emptiness_dict[column_index - 1] = non_empty_cells
                        if len(non_empty_cells) == 0:
                            left_column_empty_feature = 1
                    feature_vector.append(left_column_empty_feature)

                    # empty column right
                    right_column_empty_feature = 0
                    if column_index != table_width - 1:
                        if column_index + 1 in column_emptiness_dict:
                            non_empty_cells = column_emptiness_dict[column_index + 1]
                        else:
                            right_column = [line[column_index + 1] for line in table_annotations]
                            non_empty_cells = [cell for cell in right_column if cell != 'empty']
                            column_emptiness_dict[column_index + 1] = non_empty_cells
                        if len(non_empty_cells) == 0:
                            right_column_empty_feature = 1
                    feature_vector.append(right_column_empty_feature)

                    # derived keyword in the line
                    if row_index in derived_keyword_line_exist_dict:
                        feature_vector.append(derived_keyword_line_exist_dict[row_index])
                    else:
                        derived_keyword_line_exist_feature = 0
                        for cell in table_cells[row_index]:
                            words = [word.lower() for word in cell.split()]
                            for keyword in derived_keywords:
                                if keyword in words:
                                    derived_keyword_line_exist_feature = 1
                                    break
                            if derived_keyword_line_exist_feature == 1:
                                break
                        feature_vector.append(derived_keyword_line_exist_feature)
                        derived_keyword_line_exist_dict[row_index] = derived_keyword_line_exist_feature

                    # derived keyword in the column
                    if column_index in derived_keyword_column_exist_dict:
                        feature_vector.append(derived_keyword_column_exist_dict[column_index])
                    else:
                        derived_keyword_column_exist_feature = 0
                        column = [line[column_index] for line in table_cells]
                        for cell in column:
                            words = [word.lower() for word in cell.split()]
                            for keyword in derived_keywords:
                                if keyword in words:
                                    derived_keyword_column_exist_feature = 1
                                    break
                            if derived_keyword_column_exist_feature == 1:
                                break
                        feature_vector.append(derived_keyword_column_exist_feature)
                        derived_keyword_column_exist_dict[column_index] = derived_keyword_column_exist_feature

                    # empty cell ratio in row
                    if row_index in row_emptiness_dict:
                        non_empty_cells = row_emptiness_dict[row_index]
                    else:
                        non_empty_cells = [cell for cell in table_annotations[row_index] if cell != 'empty']
                        row_emptiness_dict[row_index] = non_empty_cells
                    empty_cell_ratio_row_feature = 1 - len(non_empty_cells) / len(table_annotations[row_index])
                    feature_vector.append(empty_cell_ratio_row_feature)

                    # empty cell ratio in column
                    if column_index in column_emptiness_dict:
                        non_empty_cells = column_emptiness_dict[column_index]
                    else:
                        this_column = [line[column_index] for line in table_annotations]
                        non_empty_cells = [cell for cell in this_column if cell != 'empty']
                        column_emptiness_dict[column_index] = non_empty_cells
                    empty_cell_ratio_column_feature = 1 - len(non_empty_cells) / table_length
                    feature_vector.append(empty_cell_ratio_column_feature)

                    # normalized block size
                    feature_vector.append(block_size[(row_index, column_index)] / table_size)

                    # whether the cell's value is aggregation of some other cells.
                    feature_vector.append(1 if (row_index, column_index) in is_sum else 0)

                    # include context information of the neighbouring cells, the used neighbours are shown as follow,
                    # 1 2 3
                    # 4 S 5
                    # 6 7 8
                    # where S is the cell itself

                    if row_index - 1 < 0 or column_index - 1 < 0:
                        position_one_value_length_feature = -1
                        position_one_data_type_feature = 5
                    else:
                        position_one_value_length_feature = table_value_length[row_index - 1][column_index - 1] / max_value_length
                        position_one_data_type_feature = table_data_types[row_index - 1][column_index - 1]
                    feature_vector.append(position_one_value_length_feature)
                    feature_vector.append(position_one_data_type_feature)

                    if row_index - 1 < 0:
                        position_two_value_length_feature = -1
                        position_two_data_type_feature = 5
                    else:
                        position_two_value_length_feature = table_value_length[row_index - 1][column_index] / max_value_length
                        position_two_data_type_feature = table_data_types[row_index - 1][column_index]
                    feature_vector.append(position_two_value_length_feature)
                    feature_vector.append(position_two_data_type_feature)

                    if row_index - 1 < 0 or column_index + 1 == table_width:
                        position_three_value_length_feature = -1
                        position_three_data_type_feature = 5
                    else:
                        position_three_value_length_feature = table_value_length[row_index - 1][column_index + 1] / max_value_length
                        position_three_data_type_feature = table_data_types[row_index - 1][column_index + 1]
                    feature_vector.append(position_three_value_length_feature)
                    feature_vector.append(position_three_data_type_feature)

                    if column_index - 1 < 0:
                        position_four_value_length_feature = -1
                        position_four_data_type_feature = 5
                    else:
                        position_four_value_length_feature = table_value_length[row_index][column_index - 1] / max_value_length
                        position_four_data_type_feature = table_data_types[row_index][column_index - 1]
                    feature_vector.append(position_four_value_length_feature)
                    feature_vector.append(position_four_data_type_feature)

                    if column_index + 1 == table_width:
                        position_five_value_length_feature = -1
                        position_five_data_type_feature = 5
                    else:
                        position_five_value_length_feature = table_value_length[row_index][column_index + 1] / max_value_length
                        position_five_data_type_feature = table_data_types[row_index][column_index + 1]
                    feature_vector.append(position_five_value_length_feature)
                    feature_vector.append(position_five_data_type_feature)

                    if row_index + 1 == table_length or column_index - 1 < 0:
                        position_six_value_length_feature = -1
                        position_six_data_type_feature = 5
                    else:
                        position_six_value_length_feature = table_value_length[row_index + 1][column_index - 1] / max_value_length
                        position_six_data_type_feature = table_data_types[row_index + 1][column_index - 1]
                    feature_vector.append(position_six_value_length_feature)
                    feature_vector.append(position_six_data_type_feature)

                    if row_index + 1 == table_length:
                        position_seven_value_length_feature = -1
                        position_seven_data_type_feature = 5
                    else:
                        position_seven_value_length_feature = table_value_length[row_index + 1][column_index] / max_value_length
                        position_seven_data_type_feature = table_data_types[row_index + 1][column_index]
                    feature_vector.append(position_seven_value_length_feature)
                    feature_vector.append(position_seven_data_type_feature)

                    if row_index + 1 == table_length or column_index + 1 == table_width:
                        position_eight_value_length_feature = -1
                        position_eight_data_type_feature = 5
                    else:
                        position_eight_value_length_feature = table_value_length[row_index + 1][column_index + 1] / max_value_length
                        position_eight_data_type_feature = table_data_types[row_index + 1][column_index + 1]
                    feature_vector.append(position_eight_value_length_feature)
                    feature_vector.append(position_eight_data_type_feature)

                    _flattened_feature_vectors.append(feature_vector)
                    _flattened_annotations.append(table_annotations[row_index][column_index])
                    _flattened_metadata.append([filename, sheetname, row_index + 1, column_index + 1])

            end = default_timer()

            runtime_create_features_this_file = str(timedelta(seconds=end - start))
            print('Elapsed time on this file: ' + runtime_create_features_this_file)
            print()
            arr_shape = np.array(table_cells).shape
            file_size = str(arr_shape[0] * arr_shape[1])

            runtime_create_features.append((identifier_file, runtime_create_features_this_file, file_size))

        print(str(timedelta(seconds=runtime_derived_cell_detection)))

        print('Average number of generated derived cell candidates: {}'.format(str(avg_num_derived_candidates / len(_all_table_cells))))

        runtime_create_cell_features_output_path = '../runtime/runtime_create_features_per_file.csv'
        headers = ['file_identifier', 'runtime in second', 'file_size']
        pandas.DataFrame(data=runtime_create_features, columns=headers).to_csv(runtime_create_cell_features_output_path, index=False)

    return _flattened_feature_vectors, _flattened_annotations, _flattened_metadata, _feature_names


def create_derived_cell_det_features(_all_table_cells, _all_table_annotations, _all_filenames, _all_sheetnames, aggregation_delta, satisfied_ratio):
    _flattened_feature_vectors = []
    _flattened_annotations = []
    _flattened_metadata = []
    _feature_names = ['is_sum']
    # , 'is_derived_ratio_row', 'is_derived_ratio_column']

    for (table_cells, table_annotations, filename, sheetname) in tqdm(zip(_all_table_cells, _all_table_annotations, _all_filenames, _all_sheetnames)):

        table_length = len(table_cells)
        table_width = 0 if table_length == 0 else len(table_cells[0])

        table_data_types = []
        for line in table_cells:
            line_data_types = [detect_datatype(cell) for cell in line]
            table_data_types.append(line_data_types)

        table_value_length = []
        for line in table_cells:
            line_value_length = [len(str.strip(cell)) for cell in line]
            table_value_length.append(line_value_length)

        num_numeric_by_row_index = {}
        for row_index in range(table_length):
            row_data_type = table_data_types[row_index]
            num_numeric_by_row_index[row_index] = len([cell_dt for cell_dt in row_data_type if cell_dt == 1 or cell_dt == 2])

        num_numeric_by_column_index = {}
        for column_index in range(table_width):
            column_data_type = [line[column_index] for line in table_data_types]
            num_numeric_by_column_index[column_index] = len([cell_dt for cell_dt in column_data_type if cell_dt == 1 or cell_dt == 2])

        is_sum = cal_is_derived(table_cells, aggregation_delta, satisfied_ratio)

        for row_index in range(table_length):
            for column_index in range(table_width):

                if table_annotations[row_index][column_index] == 'empty':
                    continue

                feature_vector = []

                # whether the cell's value is aggregation of some other cells.
                feature_vector.append(1 if (row_index, column_index) in is_sum else 0)

                # # the ratio of derived cells in the row.
                # derived_cell_indices_in_row = [derived_cell_index for derived_cell_index in is_sum if derived_cell_index[0] == row_index]
                # feature_vector.append(
                #     len(derived_cell_indices_in_row) / num_numeric_by_row_index[row_index] if num_numeric_by_row_index[row_index] != 0 else 0)
                #
                # # the ratio of derived cells in the column.
                # derived_cell_indices_in_column = [derived_cell_index for derived_cell_index in is_sum if derived_cell_index[1] == column_index]
                # feature_vector.append(len(derived_cell_indices_in_column) / num_numeric_by_column_index[column_index] if num_numeric_by_column_index[
                #                                                                                                              column_index] != 0 else 0)

                _flattened_feature_vectors.append(feature_vector)
                _flattened_annotations.append(table_annotations[row_index][column_index])
                _flattened_metadata.append([filename, sheetname, row_index + 1, column_index + 1])
    return _flattened_feature_vectors, _flattened_annotations, _flattened_metadata, _feature_names


if __name__ == '__main__':
    algorithm_type = 'cell'
    algorithm_name = 'cstrudel'

    feature_vector_path = '/Users/lan/PycharmProjects/table-content-type-classification/feature_vec'

    algorithm = cstrudel(data_folder=None, dataset_name='all', feature_vector_path=feature_vector_path)
    algorithm.train()
