import argparse
import os.path

# from create_database import load
import sys

import pandas
from pebble import ProcessPool

from strudel.cstrudel import CStrudel, create_cell_feature_vector, combine_feature_vector
from strudel.lstrudel import LStrudel, create_line_feature_vector
from strudel.classification import CrossValidation
from strudel.data import load_data
from strudel.utility import process_pebble_results

sys.path.insert(0, 'strudel')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser for line & cell classification script.')
    parser.add_argument('-d', default='saus', help="Specify the training dataset. Default 'all' uses all data-sets.")
    parser.add_argument('-t', help='Specify the test dataset. If not given, apply cross-validation on the training dataset.')
    parser.add_argument('-f', default='./data/', help='Path of the data files to be populated into the database.')
    parser.add_argument('-o', default='./result/', help='Path where all experiment result files are stored.')
    parser.add_argument('-i', default=0, type=int, help='The number of this iteration.')

    args = parser.parse_args()
    print(args)

    training_dataset = args.d
    test_dataset = args.t
    data_file_path = args.f
    output_path = os.path.join(str(args.o), training_dataset + '_result.csv')
    n_iter = str(args.i)

    max_workers = int(os.cpu_count() * 0.5)
    max_tasks = 10

    true_labels = None
    pred_labels = None
    algorithm_type = None
    description = None

    results = []
    if test_dataset is None:
        dataset_path = os.path.join(data_file_path, training_dataset + '.jl.gz')
        dataset = load_data(dataset_path=dataset_path)

        print('Creating lstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_line_feature_vectors = pool.map(create_line_feature_vector, dataset).result()

        line_fvs_dataset = process_pebble_results(optional_line_feature_vectors)

        print('Cross-validating lstrudel...')
        cv = CrossValidation(n_splits=10)
        line_cv_results = cv.cross_validate(line_fvs_dataset, LStrudel.algo_name)

        print('Creating cstrudel feature vector...')
        with ProcessPool(max_workers=max_workers, max_tasks=max_tasks) as pool:
            optional_cell_feature_vectors = pool.map(create_cell_feature_vector, dataset).result()

        cell_fvs_dataset = process_pebble_results(optional_cell_feature_vectors)

        cell_fvs_dataset = combine_feature_vector(cell_fvs_dataset, line_cv_results)
        print('Cross-validating cstrduel...')
        results = cv.cross_validate(cell_fvs_dataset, CStrudel.algorithm)

    pandas.DataFrame.to_csv(results, output_path, index=False)
