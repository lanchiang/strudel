# Created by lan at 2021/11/17
import gzip
import json

import numpy as np
import pandas

if __name__ == '__main__':
    dataset_json_path = '/Users/lan/Documents/hpi/projects/comment-detection/code/strudel/data/troy.jl.gz'
    line_label_file_path = '/Users/lan/Documents/hpi/code/line-type-classification/src/main/resources/new_annotations/annotations_troy.csv'
    output_json_path = '../enhanced_data/troy.jl.gz'

    with gzip.open(dataset_json_path) as reader:
        file_json_dicts = [json.loads(line) for line in reader]
        enhanced_dicts = {}

    for fj_dict in file_json_dicts:
        enhanced_dicts[(fj_dict['file_name'], fj_dict['table_id'])] = fj_dict

    label_df = pandas.read_csv(line_label_file_path)

    label_groupby_sheet = label_df.groupby(['FileName', 'SheetName'])

    for fj_dict in file_json_dicts:
        file_name = fj_dict['file_name']
        sheet_name = fj_dict['table_id']
        group = label_groupby_sheet.get_group((file_name, sheet_name))
        label = list(group['AnnotationLabel'])
        if len(label) != len(fj_dict['table_array']):
            print('Wrong at {}.{}'.format(file_name, sheet_name))
        fj_dict['line_annotation'] = label

    with gzip.open(output_json_path, 'wt', encoding='UTF-8') as writer:
        for fj_dict in file_json_dicts:
            writer.write(json.dumps(fj_dict) + '\n')

    pass