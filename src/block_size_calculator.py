# Created by lan at 2021/11/9
from queue import Queue

import numpy as np


def cal_block_size_linewise(table_annotations):
    table_cells_narr = np.array(table_annotations)
    marker_table = np.full_like(table_annotations, fill_value=0, dtype=int)
    marker_table[table_cells_narr != 'empty'] = -1

    indices_non_empty = np.where(marker_table == -1)
    pli_index2group = {}
    group_number_cursor = 1
    group_set = []
    blocked_index_dict = {}

    # print('Construct marker table...')

    for index_row, index_column in zip(indices_non_empty[0], indices_non_empty[1]):
        # print((index_row, index_column))
        # index of the left cell
        index_left_cell = (index_row, index_column - 1)
        left_cell_group = pli_index2group[index_left_cell] if index_left_cell in pli_index2group else []
        index_top_cell = (index_row - 1, index_column)
        top_cell_group = pli_index2group[index_top_cell] if index_top_cell in pli_index2group else []

        if not left_cell_group and not top_cell_group:
            pli_index2group[(index_row, index_column)] = [group_number_cursor]
            marker_table[(index_row, index_column)] = group_number_cursor
            group_number_cursor += 1

        else:
            groups = left_cell_group + top_cell_group
            minimal_group_number = min(groups)
            marker_table[(index_row, index_column)] = minimal_group_number
            pli_index2group[(index_row, index_column)] = [minimal_group_number]
            if np.unique(groups).size > 1:
                group_set.append(groups)

    group_set = set(tuple(i) for i in group_set)
    # print('Merge groups in the marker table...')

    for group in group_set:
        minimal_group_number = min(group)
        for group_id in group:
            marker_table[marker_table == group_id] = minimal_group_number

    (unique, counts) = np.unique(marker_table, return_counts=True)

    # print('Assign block size to all indices...')
    for group_id, count in zip(unique, counts):
        if group_id == 0:
            continue
        indices = np.where(marker_table == group_id)
        for index_row, index_column in zip(indices[0], indices[1]):
            blocked_index_dict[(index_row, index_column)] = count
        # print('Found block size for ' + str(count) + ' cells.')

    return blocked_index_dict


def cal_block_size(table_annotations):
    table_cells_narr = np.array(table_annotations)
    marker_table = np.full_like(table_annotations, fill_value=0, dtype=int)
    marker_table[table_cells_narr != 'empty'] = 1

    indices_non_empty = np.where(marker_table == 1)
    indices_list = []
    for index_row, index_column in zip(indices_non_empty[0], indices_non_empty[1]):
        indices_list.append((index_row, index_column))

    blocked_index_dict = {}
    while len(indices_list) != 0:
        bs = 1
        first_index = indices_list[0]
        queue = Queue()
        queue.put(first_index)
        marker_table[first_index] = 0
        indices_list.remove(first_index)
        blocked_index_list = []
        while not queue.empty():
            index = queue.get()
            print(index)
            blocked_index_list.append(index)
            # top
            index_top_cell = (index[0] - 1, index[1])
            if marker_table[index_top_cell] == 1 and index_top_cell[0] >= 0:
                marker_table[index_top_cell] = 0
                queue.put(index_top_cell)
                indices_list.remove(index_top_cell)
                bs +=1
            # bottom
            index_bottom_cell = (index[0] + 1, index[1])
            if index_bottom_cell[0] < marker_table.shape[0]:
                if marker_table[index_bottom_cell] == 1:
                    marker_table[index_bottom_cell] = 0
                    queue.put(index_bottom_cell)
                    indices_list.remove(index_bottom_cell)
                    bs += 1
            # left
            index_left_cell = (index[0], index[1] - 1)
            if marker_table[index_left_cell] == 1 and index_left_cell[1] >= 0:
                marker_table[index_left_cell] = 0
                queue.put(index_left_cell)
                indices_list.remove(index_left_cell)
                bs += 1
            # right
            index_right_cell = (index[0], index[1] + 1)
            if index_right_cell[1] < marker_table.shape[1]:
                if marker_table[index_right_cell] == 1:
                    marker_table[index_right_cell] = 0
                    queue.put(index_right_cell)
                    indices_list.remove(index_right_cell)
                    bs += 1
        print('Found block size for ' + str(len(blocked_index_list)) + ' cells.')
        _dict = dict(zip(blocked_index_list, [bs] * len(blocked_index_list)))
        blocked_index_dict.update(_dict)

    return blocked_index_dict
