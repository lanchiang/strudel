# Created by lan at 2021/11/17
import pandas
from sklearn.model_selection import StratifiedKFold

from strudel.cstrudel import CStrudel
from strudel.lstrudel import LStrudel


class CrossValidation:

    def __init__(self, n_splits=10, shuffle=True):
        self.n_splits = n_splits
        self.shuffle = shuffle

    def cross_validate(self, dataset: pandas.DataFrame, model):
        profile_columns = ['file_name', 'sheet_name']

        clean_dataset = dataset[dataset['label'] != 'empty'].reset_index(drop=True)

        if not isinstance(clean_dataset, pandas.DataFrame):
            clean_dataset = pandas.DataFrame(clean_dataset)

        groups_by_file = clean_dataset.groupby(profile_columns)
        file_names = list(groups_by_file.groups.keys())

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)

        results = []
        for train_indices, test_indices in skf.split(file_names, [0] * len(file_names)):
            file_names_train_set = [file_names[index] for index in train_indices]
            groups = []
            for file_name in file_names_train_set:
                group = groups_by_file.get_group(file_name)
                groups.append(group)

            train_df = pandas.concat(groups, axis=0)

            file_names_test_set = [file_names[index] for index in test_indices]
            groups = []
            for file_name in file_names_test_set:
                group = groups_by_file.get_group(file_name)
                groups.append(group)

            test_df = pandas.concat(groups, axis=0)

            if model == 'lstrudel':
                ls = LStrudel()
                result = ls.fit(train_df, test_df)
                results.append(result)
            elif model == 'cstrudel':
                cs = CStrudel()
                result = cs.fit(train_df, test_df)
                results.append(result)

        cv_result_df = pandas.concat(results, axis=0)

        return cv_result_df
