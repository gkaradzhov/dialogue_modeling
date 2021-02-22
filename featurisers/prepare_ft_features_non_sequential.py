import csv

from sklearn.model_selection import LeaveOneOut

from featurisers.raw_wason_featuriser import get_y
from outcome_prediction.linear_pipeline import FEATURE_MAPS
from outcome_prediction.prediction_utils import get_features, merge_feauters, features_labels_to_xy
from read_data import read_wason_dump
import numpy as np


if __name__ == '__main__':

    sc_turns = get_features(FEATURE_MAPS, 'street_crowd_turns')
    sc_messages = get_features(FEATURE_MAPS, 'street_crowd_messages')
    meta_feats = get_features(FEATURE_MAPS, 'dialogue_metadata')
    sol_part = get_features(FEATURE_MAPS, 'solution_participation_automatic')
    annotation = get_features(FEATURE_MAPS, 'annotation_features')

    merged_feats_dict = merge_feauters([meta_feats, sol_part])
    raw_data = read_wason_dump('../data/all/')

    Y = get_y(raw_data)

    processed = []
    for key, item in merged_feats_dict.items():
        if key in Y:
            processed.append([key, *item])
    merged_feats = np.array(processed)
    loo = LeaveOneOut()

    id = 0
    for train_index, test_index in loo.split(merged_feats):
        fname_train = '../features/fast_text/stat_train_{}.txt'.format(id)
        fname_test = '../features/fast_text/stat_test_{}.tsv'.format(id)
        id += 1
        X_train, X_test = merged_feats[train_index], merged_feats[test_index]

        with open(fname_train, 'w') as wf:
            for item in X_train:
                y = Y[item[0]]
                wf.write('__label__' + str(y) + ' ' + " ".join(item[1:]) + '\n')

        with open(fname_test, 'w') as wf:
            tsv_writer = csv.writer(wf, delimiter='\t')
            for item in X_test:
                y = Y[item[0]]
                tsv_writer.writerow([
                    str(item[0]),
                    str(y),
                    " ".join(item[1:])])