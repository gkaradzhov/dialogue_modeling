import csv
from copy import copy
import numpy as np
from datetime import datetime

def get_features(feature_map, feature_name):
    feature_name = feature_map[feature_name]

    feature_dict = {}
    with open(feature_name) as rf:
        csv_reader = csv.reader(rf, delimiter='\t')
        for item in csv_reader:
            feature_dict[item[0]] = [float(i) for i in item[1:]]

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

    return np.array(X_new), np.array(Y_new)


def logging(fname, feature_names, pipeline_config, score):
    with open(fname, 'a+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        csv_writer.writerow([str(datetime.now()),
                               ",".join(feature_names),
                               str(pipeline_config),
                               str(score)])