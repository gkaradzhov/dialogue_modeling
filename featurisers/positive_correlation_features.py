import csv
import json
from collections import Counter
import ast

import os
import pandas as pd

from featurisers.raw_wason_featuriser import preprocess_conversation_dump, calculate_stats, get_y
from read_data import read_solution_annotaions, read_wason_dump, read_3_lvl_annotation_file


def get_wason_feats(collection):
    stats = []
    for d in collection:
        try:
            prepr = preprocess_conversation_dump(d.raw_db_conversation)
            s = calculate_stats(prepr)

            if s['num_of_playing_wason'] >= 2:
                s['identifier'] = d.identifier
                stats.append(s)
        except Exception as e:
            print(e)

    result = {}
    for stat in stats:
        result[stat['identifier']] = stat['message_count']

    return result


def init_feature_map():
    return {
        'Probing': 0,
        'Probing_Reasoning': 0,
        'Probing_Solution': 0,
        'Probing_Moderation': 0,
        'Non-probing-deliberation': 0,
        'Non-probing-deliberation_Solution': 0,
        'Non-probing-deliberation_Reasoning': 0,
        'Non-probing-deliberation_Agree': 0,
        'Non-probing-deliberation_Disagree': 0,
        'Reasoning': 0,
        'Solution': 0,
        'Agree': 0,
        'Disagree': 0,
        'Moderation': 0,
        'complete_solution': 0,
        'partial_solution': 0,
        'consider_opposite': 0,
        'solution_summary': 0,
        'specific_addressee': 0,
        '0': 0
    }


def annotation_features(conversation_collection):
    features = {}
    for a in conversation_collection:
        f_id = a.identifier
        fmap = init_feature_map()

        for m in a.wason_messages:
            annotation_obj = m.annotation
            if annotation_obj['type'] == '0':
                continue
            fmap[annotation_obj['type']] += 1
            fmap[annotation_obj['target']] += 1
            key = "{}_{}".format(annotation_obj['type'], annotation_obj['target'])
            fmap[key] += 1

            for item in annotation_obj['additional']:
                fmap[item] += 1

        normalised = []
        for key, item in fmap.items():
            normalised.append(float(item / len(a.wason_messages)))
        fmap['has_3_probing'] = 1 if fmap['Probing'] >= 3 else 0
        fmap['has_3_probing_solution'] = 1 if fmap['Probing_Solution'] >= 3 else 0
        fmap['has_3_probing_reasoning'] = 1 if fmap['Probing_Reasoning'] >= 3 else 0

        features[f_id] = [fmap['Probing_Reasoning'] / len(a.wason_messages),
                          fmap['Solution'] / len(a.wason_messages)]

    return features


if __name__ == '__main__':
    raw_data = read_wason_dump('../data/all/')
    wason_feats = get_wason_feats(raw_data)

    anns = read_3_lvl_annotation_file('../3lvl_anns.tsv')

    annotation = annotation_features(anns)
    Y = get_y(raw_data)

    with open('../features/positive_correlation_features_with_Y.tsv', 'w') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for key, message_len in wason_feats.items():
            if key in annotation:
                ann = annotation[key]
                y = Y[key]
                csv_writer.writerow([key, message_len, *ann, y])


