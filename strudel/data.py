# Created by lan at 2021/11/12
import gzip
import json

import numpy as np


def load_data(dataset_path):
    with gzip.open(dataset_path, mode='r') as ds_json_file:
        json_file_dicts = np.array([json.loads(line) for line in ds_json_file])
    return json_file_dicts