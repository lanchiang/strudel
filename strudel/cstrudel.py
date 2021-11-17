# 
# Created by lan at 2020/3/3

import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier

from strudel.block_size_calculator import cal_block_size_linewise
from strudel.derived_detector import cal_is_derived
from strudel.utility import detect_datatype


def combine_feature_vector(cell_feature_vector: pandas.DataFrame, line_feature_vector: pandas.DataFrame):
    cell_fv_df = line_feature_vector.drop(['label'], axis=1).merge(cell_feature_vector,
                                                                   left_on=['file_name', 'sheet_name', 'line_number'],
                                                                   right_on=['file_name', 'sheet_name', 'row_index'])
    cell_fv_df = cell_fv_df.drop(['line_number'], axis=1)
    return cell_fv_df


class CStrudel:
    algorithm = 'cstrudel'

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, train_set: pandas.DataFrame, test_set: pandas.DataFrame):
        profile_columns = ['file_name', 'sheet_name', 'row_index', 'column_index']

        test_set_cell_profile = test_set[profile_columns]

        clean_train_set = train_set.drop(profile_columns, axis=1)
        clean_test_set = test_set.drop(profile_columns, axis=1)

        X_train = clean_train_set.iloc[:, 0:len(clean_train_set.columns) - 1]
        y_train = clean_train_set.iloc[:, len(clean_train_set.columns) - 1:len(clean_train_set.columns)]

        X_test = clean_test_set.iloc[:, 0:len(clean_test_set.columns) - 1]
        y_test = clean_test_set.iloc[:, len(clean_test_set.columns) - 1: len(clean_test_set.columns)]

        clf = RandomForestClassifier(n_jobs=self.n_jobs)

        clf.fit(X_train, np.ravel(y_train))

        pred = pandas.DataFrame(clf.predict(X_test), columns=['predict'], index=y_test.index)

        result = pandas.concat([test_set_cell_profile, pred, y_test], axis=1)

        return result


def create_cell_feature_vector(file_json_dict):
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

    file_name = file_json_dict['file_name']
    sheet_name = file_json_dict['table_id']

    file_content = file_json_dict['table_array']
    cell_annotations = file_json_dict['annotations']

    block_size = cal_block_size_linewise(cell_annotations)

    file_length = file_json_dict['num_rows']
    file_width = file_json_dict['num_cols']
    file_size = file_length * file_width

    max_value_length = max([len(str.strip(cell)) for file_line in file_content for cell in file_line])

    table_data_types = []
    for line in file_content:
        line_data_types = [detect_datatype(cell) for cell in line]
        table_data_types.append(line_data_types)

    table_value_length = []
    for line in file_content:
        line_value_length = [len(str.strip(cell)) for cell in line]
        table_value_length.append(line_value_length)

    num_numeric_by_row_index = {}
    for row_index in range(file_length):
        row_data_type = table_data_types[row_index]
        num_numeric_by_row_index[row_index] = len([cell_dt for cell_dt in row_data_type if cell_dt == 1 or cell_dt == 2])

    num_numeric_by_column_index = {}
    for column_index in range(file_width):
        column_data_type = [line[column_index] for line in table_data_types]
        num_numeric_by_column_index[column_index] = len([cell_dt for cell_dt in column_data_type if cell_dt == 1 or cell_dt == 2])

    row_emptiness_dict = {}
    column_emptiness_dict = {}

    derived_keyword_line_exist_dict = {}
    derived_keyword_column_exist_dict = {}

    is_sum, num_derived_candidates = cal_is_derived(file_content, aggr_delta=1.0E-1)
    # avg_num_derived_candidates += num_derived_candidates

    _flattened_feature_vectors = []
    _flattened_annotations = []
    _flattened_metadata = []

    # print('Calculate features for cells...')
    for row_index in range(file_length):
        for column_index in range(file_width):

            if cell_annotations[row_index][column_index] == 'empty' or cell_annotations[row_index][column_index] is None:
                continue

            feature_vector = []

            # normalized row position
            row_position_feature = row_index / file_length
            feature_vector.append(row_position_feature)

            # normalized column position
            column_position_feature = column_index / file_width
            feature_vector.append(column_position_feature)

            # value length
            value_length_feature = len(str.strip(file_content[row_index][column_index])) / max_value_length
            feature_vector.append(value_length_feature)

            # data type
            value_data_type_feature = detect_datatype(file_content[row_index][column_index])
            feature_vector.append(value_data_type_feature)

            # is empty line above
            above_line_empty_feature = 0
            if row_index != 0:
                if row_index - 1 in row_emptiness_dict:
                    non_empty_cells = row_emptiness_dict[row_index - 1]
                else:
                    non_empty_cells = [cell for cell in cell_annotations[row_index - 1] if cell != 'empty']
                    row_emptiness_dict[row_index - 1] = non_empty_cells
                if len(non_empty_cells) == 0:
                    above_line_empty_feature = 1
            feature_vector.append(above_line_empty_feature)

            # is empty line after
            after_line_empty_feature = 0
            if row_index != file_length - 1:
                if row_index + 1 in row_emptiness_dict:
                    non_empty_cells = row_emptiness_dict[row_index + 1]
                else:
                    non_empty_cells = [cell for cell in cell_annotations[row_index + 1] if cell != 'empty']
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
                    left_column = [line[column_index - 1] for line in cell_annotations]
                    non_empty_cells = [cell for cell in left_column if cell != 'empty']
                    column_emptiness_dict[column_index - 1] = non_empty_cells
                if len(non_empty_cells) == 0:
                    left_column_empty_feature = 1
            feature_vector.append(left_column_empty_feature)

            # empty column right
            right_column_empty_feature = 0
            if column_index != file_width - 1:
                if column_index + 1 in column_emptiness_dict:
                    non_empty_cells = column_emptiness_dict[column_index + 1]
                else:
                    right_column = [line[column_index + 1] for line in cell_annotations]
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
                for cell in file_content[row_index]:
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
                column = [line[column_index] for line in file_content]
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
                non_empty_cells = [cell for cell in cell_annotations[row_index] if cell != 'empty']
                row_emptiness_dict[row_index] = non_empty_cells
            empty_cell_ratio_row_feature = 1 - len(non_empty_cells) / len(cell_annotations[row_index])
            feature_vector.append(empty_cell_ratio_row_feature)

            # empty cell ratio in column
            if column_index in column_emptiness_dict:
                non_empty_cells = column_emptiness_dict[column_index]
            else:
                this_column = [line[column_index] for line in cell_annotations]
                non_empty_cells = [cell for cell in this_column if cell != 'empty']
                column_emptiness_dict[column_index] = non_empty_cells
            empty_cell_ratio_column_feature = 1 - len(non_empty_cells) / file_length
            feature_vector.append(empty_cell_ratio_column_feature)

            # normalized block size
            feature_vector.append(block_size[(row_index, column_index)] / file_size)

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

            if row_index - 1 < 0 or column_index + 1 == file_width:
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

            if column_index + 1 == file_width:
                position_five_value_length_feature = -1
                position_five_data_type_feature = 5
            else:
                position_five_value_length_feature = table_value_length[row_index][column_index + 1] / max_value_length
                position_five_data_type_feature = table_data_types[row_index][column_index + 1]
            feature_vector.append(position_five_value_length_feature)
            feature_vector.append(position_five_data_type_feature)

            if row_index + 1 == file_length or column_index - 1 < 0:
                position_six_value_length_feature = -1
                position_six_data_type_feature = 5
            else:
                position_six_value_length_feature = table_value_length[row_index + 1][column_index - 1] / max_value_length
                position_six_data_type_feature = table_data_types[row_index + 1][column_index - 1]
            feature_vector.append(position_six_value_length_feature)
            feature_vector.append(position_six_data_type_feature)

            if row_index + 1 == file_length:
                position_seven_value_length_feature = -1
                position_seven_data_type_feature = 5
            else:
                position_seven_value_length_feature = table_value_length[row_index + 1][column_index] / max_value_length
                position_seven_data_type_feature = table_data_types[row_index + 1][column_index]
            feature_vector.append(position_seven_value_length_feature)
            feature_vector.append(position_seven_data_type_feature)

            if row_index + 1 == file_length or column_index + 1 == file_width:
                position_eight_value_length_feature = -1
                position_eight_data_type_feature = 5
            else:
                position_eight_value_length_feature = table_value_length[row_index + 1][column_index + 1] / max_value_length
                position_eight_data_type_feature = table_data_types[row_index + 1][column_index + 1]
            feature_vector.append(position_eight_value_length_feature)
            feature_vector.append(position_eight_data_type_feature)

            _flattened_feature_vectors.append(feature_vector)
            _flattened_annotations.append(cell_annotations[row_index][column_index])
            _flattened_metadata.append([file_name, sheet_name, row_index, column_index])

    feature_vector_df = pandas.DataFrame(_flattened_feature_vectors, columns=_feature_names)
    cell_label_df = pandas.DataFrame(_flattened_annotations, columns=['label'])
    cell_profile_df = pandas.DataFrame(_flattened_metadata, columns=['file_name', 'sheet_name', 'row_index', 'column_index'])

    cell_feature_vector = pandas.concat([cell_profile_df, feature_vector_df, cell_label_df], axis=1)

    return cell_feature_vector
