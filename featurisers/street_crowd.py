import csv

import spacy

from external_tools.cornellversation.constructive.msg_features import message_features
from external_tools.cornellversation.constructive.turn_features import turn_features, turns_from_chat

from read_data import read_wason_dump

def featurise(path_message, path_turns):
    raw = read_wason_dump('../data/all_data_20210107/')

    nlp = spacy.load("en_core_web_sm")
    for item in raw:
        item.wason_messages_from_raw()
        item.preprocess_everything(nlp)

    sc_format = []
    for conv in raw:
        sc_format.append(conv.to_street_crowd_format())

    message_fs = []
    turn_feats = []
    for ann, conversation in zip(raw, sc_format):
        mf_all = message_features(conversation)

        mf = mf_all[0]
        mf_av = average_dict(mf, mf[0].keys())
        message_fs.append([ann.identifier, *mf_av])

        all_t = turn_features(turns_from_chat(conversation))
        tf_av = average_dict(all_t, ('agree', 'disagree', 'n_repeated_content',
                                     'n_repeated_stop', 'n_repeated_pos_bigram', 'gap'))
        turn_feats.append([ann.identifier, *tf_av])

    with open(path_message, 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in message_fs:
            csv_writer.writerow(item)

    with open(path_turns, 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in turn_feats:
            csv_writer.writerow(item)

def average_dict(collection, keys):
    averaged = dict.fromkeys(keys, 0.0)

    if len(collection) >= 1:
        for item in collection:
            for k in averaged.keys():
                averaged[k] += item.get(k, 0.0)

        for k, v in averaged.items():
            averaged[k] = v / len(collection)

    return averaged.values()



if __name__ == '__main__':

    featurise('../features/sc_messages.tsv', '../features/sc_turns.tsv')


