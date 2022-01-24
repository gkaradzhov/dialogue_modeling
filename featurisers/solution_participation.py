import csv
from copy import copy

import spacy
import string
from read_data import read_solution_annotaions, read_wason_dump, read_3_lvl_annotation_file
from supporting_classifiers.agreement_classifier import Predictor
from wason_message import WasonConversation, WasonMessage
import pandas as pd
from solution_tracker.simple_sol import process_raw_to_solution_tracker, solution_tracker
from collections import defaultdict


def featurise_solution_participation(solutions, conversations, path):
    features = {}
    for k_sol, sol in solutions.items():
        solution_changes = 0
        participation_at_20 = {}
        participation_at_30 = {}

        wason_conv = [a for a in conversations if a.identifier == k_sol][0]
        latest_sol = {}
        participation_tracker = {}
        participation_normalised = {}
        messages = 0
        for item in sol:
            if item['type'] == 'INITIAL':
                latest_sol[item['user']] = " ".join(item['value'])
            else:
                if item['user'] not in latest_sol:
                    latest_sol[item['user']] = 'N/A'

        for user in latest_sol.keys():
            participation_normalised[user + '_participation'] = 0
            participation_tracker[user] = 0

        for raw in wason_conv.raw_db_conversation:
            if raw['user_status'] != 'USR_PLAYING':
                continue

            if raw['message_type'] == 'WASON_SUBMIT':
                local_sol = [s for s in sol if s['id'] == raw['message_id']][0]
                norm_value = " ".join(local_sol['value'])
            elif raw['message_type'] == 'CHAT_MESSAGE':
                local_sol = [s for s in sol if s['id'] == raw['message_id']]
                if len(local_sol) == 0:
                    local_sol = {'user': raw['user_name'], 'value': latest_sol[raw['user_name']]}
                    norm_value = latest_sol[raw['user_name']]
                else:
                    local_sol = local_sol[0]
                    norm_value = " ".join(local_sol['value'])

                messages += 1
                participation_tracker[raw['user_name']] += 1
                for k_abs, v_abs in participation_tracker.items():
                    participation_normalised[k_abs + '_participation'] = v_abs / messages

                if messages == 20:
                    participation_at_20 = participation_normalised
                if messages == 30:
                    participation_at_30 = participation_normalised
            else:
                continue

            if norm_value != latest_sol[local_sol['user']]:
                solution_changes += 1

            latest_sol[local_sol['user']] = norm_value

        features[k_sol] = [solution_changes / messages,
                           # solution_changes / len(participation_tracker),
                           *create_participation_feats(participation_at_20),
                           *create_participation_feats(participation_at_30),
                           *create_participation_feats(participation_normalised),
                           ]

    with open(path, 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for key, stat in features.items():
            csv_writer.writerow([key, *stat])


def create_participation_feats(participation):
    if len(participation) == 0:
        return [0, 0, 0, 0]

    dominating_50 = 0
    dominating_40 = 0
    completely_silent_participant = 0
    moderatly_silent = 0

    for us, part in participation.items():
        if part >= 0.5:
            dominating_50 = 1
        elif part >= 0.4:
            dominating_40 = 1
        elif part == 0:
            completely_silent_participant = 1
        elif part >= 0 and part <= 0.2:
            moderatly_silent = 1

    return [dominating_50, dominating_40, completely_silent_participant, moderatly_silent]


if __name__ == '__main__':
    # anns = read_solution_annotaions('../solution_annotations.tsv')
    nlp = spacy.load("en_core_web_sm")
    # for a in anns:
    #     a.preprocess_everything(nlp)

    agreement_predictor = Predictor('../models/agreement.pkl')

    raw_data = read_wason_dump('../data/final_all/')

    # hierch_data = read_3_lvl_annotation_file('../3lvl_anns.tsv')

    conversations_to_process = []
    # for conv in hierch_data:
    #     raw = [d for d in raw_data if d.identifier == conv.identifier][0]
    #     conv.raw_db_conversation = raw.raw_db_conversation
    #     conversations_to_process.append(conv)

    for item in raw_data:
        item.wason_messages_from_raw()
        item.preprocess_everything(nlp)

    sols = defaultdict(lambda x: [])
    for conv in raw_data:
        sol_tracker = solution_tracker(conv, False, None)
        sols[conv.identifier] = sol_tracker

    featurise_solution_participation(sols, raw_data, '../features/solution_participation.tsv')
