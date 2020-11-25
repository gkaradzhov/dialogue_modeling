import os
import csv

FEATURE_MAPS = {
    'street_crowd': '/',
    'annotation_features': '/',
    'dialogue_metadata': '/',
    'text_features': '/'
}


def get_features(feature_name):
    fname = FEATURE_MAPS[feature_name]

    feature_dict = {}
    with open(fname):
        csv_reader = csv.reader(fname, delimiter='\t')
        for item in csv_reader:
            feature_dict[item[0]] = item[1:]

    return feature_dict


def merge_feauters(feature_map):
    merged = {}
    for feature in feature_map:
        for key, item in feature.items():
            if key in merged:
                merged[key].extend(item)
            else:
                merged[key] = item

    return merged


def features_labels_to_xy(features, labels):
    X = []
    Y = []

    for f_id, feats in features.items():
        X.append(feats)
        Y.append(labels[f_id])

    return X, Y


def logging(feature_names, pipeline_config, score):
    pass

if __name__ == '__main__':


    # 1. Read Labels

    # 2. Get features

    # 3. Normalise everything in format

    # 4. Create pipeline

    # 5. Parameter tuning

    # 6. LOOCV

    # 7. Record

    pass
