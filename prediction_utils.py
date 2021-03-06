import csv
from copy import copy
import numpy as np
from datetime import datetime
import os
from collections import defaultdict, Counter
import statistics


def get_features(feature_map, feature_name):
    feature_name = feature_map[feature_name]

    feature_dict = {}
    with open(feature_name) as rf:
        csv_reader = csv.reader(rf, delimiter='\t')
        for item in csv_reader:
            feature_dict[item[0]] = [float(i) for i in item[1:]]

    return feature_dict


def get_raw_features(path):
    feature_dict = {}
    with open(path) as rf:
        csv_reader = csv.reader(rf, delimiter='\t')
        for item in csv_reader:
            feature_dict[item[0]] = [i for i in item[1:]]

    return feature_dict


def merge_feauters(feature_map):
    merged = {}
    for feature in feature_map:
        for key, item in feature.items():
            if key in merged:
                merged[key].extend(item)
            else:
                merged[key] = copy(item)

    return merged


def features_labels_to_xy(features, labels, allowed_ids=None):
    X_new = []
    Y_new = []

    for f_id, feats in features.items():
        if allowed_ids and f_id not in allowed_ids:
            continue
        X_new.append(feats)
        Y_new.append(labels[f_id])

    return X_new, np.array(Y_new)


def features_to_arrays(features, labels):
    return_X_dict = defaultdict(lambda: [])
    Y_new = []

    id_orderring = []
    for feature_key, features in features.items():
        if len(id_orderring) == 0:
            for id_, feats in features.items():
                id_orderring.append(id_)
                return_X_dict[feature_key].append(feats)
        else:
            for item in id_orderring:
                return_X_dict[feature_key].append(features[item])

    for item in id_orderring:
        Y_new.append(labels[item])

    return return_X_dict, np.array(Y_new), id_orderring


def logging(fname, feature_names, pipeline_config, score, additional_information=''):
    with open(fname, 'a+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        csv_writer.writerow([str(datetime.now()),
                             str(score),
                             ",".join(feature_names),
                             str(pipeline_config),
                             additional_information
                             ])


def read_folder_features(path):
    feats = {}

    fnames = os.listdir(path)
    for fname in fnames:
        identifier = fname.split('.')[0]
        with open(path + fname, 'r') as rf:
            csv_reader = csv.reader(rf, delimiter='\t')
            list_data = []
            for row in csv_reader:
                list_data.append(np.array([float(a) for a in row]))
            feats[identifier] = np.array(list_data, dtype=np.float)
    return feats


def decision_tree_representation(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    repres = {
        'first_level_feat': None,
        'first_level_thresh': None,
        'second_level_feats': set()
    }

    for i in range(n_nodes):
        if not is_leaves[i]:
            if node_depth[i] == 0:
                repres['first_level_feat'] = feature[i]
                repres['first_level_thresh'] = threshold[i]
            elif node_depth[i] == 1:
                repres['second_level_feats'].update(str(feature[i]))
    return repres

def decision_tree_stats(stat_array):
    fl_counter = Counter([s['first_level_feat'] for s in stat_array])
    fl_stats = fl_counter.most_common(1)[0][1] / len(stat_array)
    mean = statistics.mean([s['first_level_thresh'] for s in stat_array])
    variance = statistics.variance([s['first_level_thresh'] for s in stat_array])

    sl_counter = Counter([str(s['second_level_feats']) for s in stat_array])
    sl_stats = sl_counter.most_common(1)[0][1] / len(stat_array)

    return {'firs_level': fl_stats, 'first_level_thresh': mean, 'first_level_var': variance,
            'second_level': sl_stats}