import json
from collections import Counter
import ast

import os
import pandas as pd

from read_data import read_solution_annotaions, read_wason_dump

allowed = {
    'vowels': {'A', 'O', 'U', 'E', 'I', 'Y'},
    'consonants': {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'X', 'Z', 'W'},
    'odds': {'1', '3', '5', '7', '9'},
    'evens': {'0', '2', '4', '6', '8'}
}


# all cards with vowels on one side have even numbers on the other
# To prove the rule, one should turn the vowel and the odd number
def is_solution_absolute(state):

    for item in state:
        if (item['value'] in allowed['vowels'] or item['value'] in allowed['odds']):
            if item['checked'] is False:
                return (0, 'WRONG')
            else:
                continue
        elif item['checked'] is True:
            return (0, 'WRONG')
    return (1, 'CORRECT')


def is_solution_fine_grained(state):
    checked_dict = {'vowels': False, 'consonants': False, 'odds': False, 'evens': False}

    if isinstance(state, list):
        for item in state:
            for checked in checked_dict.keys():
                if item['value'] in allowed[checked] and item['checked'] is True:
                    checked_dict[checked] = True
    else:
        for item in state:
            for checked in checked_dict.keys():
                if item in allowed[checked]:
                    checked_dict[checked] = True

    score = 0
    classification = 'OTHER_ERROR'
    for key, item in checked_dict.items():
        if key == 'vowels' and item:
            score += 0.25
        elif key == 'odds' and item:
            score += 0.25
        elif key == 'consonants' and not item:
            score += 0.25
        elif key == 'evens' and not item:
            score += 0.25

    if score == 1:
        classification = 'CORRECT'
    elif score == 0:
        classification = 'ALL_INCORRECT'

    if checked_dict['vowels'] and checked_dict['evens']:
        if checked_dict['odds']:
            classification = 'BIASED + ODDS'
        elif checked_dict['consonants']:
            classification = 'BIASED + CONST'
        else:
            classification = 'BIASED'

    if all(value for value in checked_dict.values()):
        classification = 'ALL_CHECKED'

    if all(value is False for value in checked_dict.values()):
        classification = 'NONE_CHECKED'

    return score, classification


import csv
from collections import defaultdict


def preprocess_conversation_dump(raw_data):

    user_performance = defaultdict(
    lambda: {'user_name': '', 'ONBOARDING_CLICK': [], 'ONBOARDING_CLICK_COARSE': [], 'ONBOARDING_SUBMIT': [],
             'ONBOARDING_SUBMIT_COARSE': [], 'GAME_CLICK': [], 'GAME_CLICK_COARSE': [],
             'GAME_SUBMIT': [], 'GAME_SUBMIT_COARSE': [], 'ONBOARDING_FINAL': '',
             'SUBMIT_FINAL': '', 'MESSAGES_TOKENIZED': [], 'user_type': ''})

    for item in raw_data:
        user_performance[item['user_id']]['user_name'] = item['user_name']
        user_performance[item['user_id']]['user_type'] = item['user_type']
        if item['user_status'] == 'USR_ONBOARDING' and item['message_type'] == 'WASON_GAME':
            user_performance[item['user_id']]['ONBOARDING_CLICK'].append(is_solution_fine_grained(item['content']))
            user_performance[item['user_id']]['ONBOARDING_CLICK_COARSE'].append(is_solution_absolute(item['content']))
        elif item['user_status'] == 'USR_ONBOARDING' and item['message_type'] == 'WASON_SUBMIT':
            user_performance[item['user_id']]['ONBOARDING_SUBMIT'].append(is_solution_fine_grained(item['content']))
            user_performance[item['user_id']]['ONBOARDING_SUBMIT_COARSE'].append(is_solution_absolute(item['content']))
            user_performance[item['user_id']]['ONBOARDING_FINAL'] = item['content']
        elif item['user_status'] == 'USR_PLAYING' and item['message_type'] == 'WASON_GAME':
            user_performance[item['user_id']]['GAME_CLICK'].append(is_solution_fine_grained(item['content']))
            user_performance[item['user_id']]['GAME_CLICK_COARSE'].append(is_solution_absolute(item['content']))

        elif item['user_status'] == 'USR_PLAYING' and item['message_type'] == 'WASON_SUBMIT':
            user_performance[item['user_id']]['GAME_SUBMIT'].append(is_solution_fine_grained(item['content']))
            user_performance[item['user_id']]['GAME_SUBMIT_COARSE'].append(is_solution_absolute(item['content']))

            user_performance[item['user_id']]['SUBMIT_FINAL'] = item['content']
        elif item['message_type'] == 'CHAT_MESSAGE':
            user_performance[item['user_id']]['MESSAGES_TOKENIZED'].append(item['content'].lower().split(' '))

    to_del = set()

    #     print(user_performance)

    for key, values in user_performance.items():
        if len(values['MESSAGES_TOKENIZED']) < 2 or values['user_name'] == 'Moderating Owl':
            to_del.add(key)

        if len(values['ONBOARDING_CLICK']) == 0:
            values['ONBOARDING_CLICK'] = [(0, 'None')]
            values['ONBOARDING_CLICK_COARSE'] = [(0, 'None')]

        if len(values['ONBOARDING_SUBMIT']) == 0:
            values['ONBOARDING_SUBMIT'] = [values['ONBOARDING_CLICK'][-1]]
            values['ONBOARDING_SUBMIT_COARSE'] = [values['ONBOARDING_CLICK_COARSE'][-1]]

        if len(values['GAME_CLICK']) == 0:
            values['GAME_CLICK'] = [values['ONBOARDING_SUBMIT'][-1]]
            values['GAME_CLICK_COARSE'] = [values['ONBOARDING_SUBMIT_COARSE'][-1]]

        if len(values['GAME_SUBMIT']) == 0:
            values['GAME_SUBMIT'] = [values['GAME_CLICK'][-1]]
            values['GAME_SUBMIT_COARSE'] = [values['GAME_CLICK_COARSE'][-1]]

    for td in to_del:
        del user_performance[td]

    return user_performance


import string

table = str.maketrans(dict.fromkeys(string.punctuation))


def calculate_stats(conversations_dump):
    result_stats = {
        'onboarding_score': 0,
        'onboarding_score_coarse': 0,
        'game_score': 0,
        'game_score_coarse': 0,
        'message_count': 0,
        'tokens_count': 0,
        'unique_tokens': Counter(),
        'onboarding_types': Counter(),
        'game_types': Counter(),
        'number_of_submits': 0
    }

    onboarding_versions = []
    final_versions = []
    for _, user in conversations_dump.items():
        result_stats['onboarding_score'] += user['ONBOARDING_SUBMIT'][-1][0]
        result_stats['onboarding_score_coarse'] += user['ONBOARDING_SUBMIT_COARSE'][-1][0]

        result_stats['game_score'] += user['GAME_SUBMIT'][-1][0]
        result_stats['game_score_coarse'] += user['GAME_SUBMIT_COARSE'][-1][0]

        checked_onbording = [a['value'] for a in user['ONBOARDING_FINAL'] if a['checked'] is True]
        if len(checked_onbording) == 0:
            checked_onbording = ['None', 'None']

        checked_final = [a['value'] for a in user['SUBMIT_FINAL'] if a['checked'] is True]
        if len(checked_final) == 0:
            checked_final = ['None', 'None']

        if user['user_type'] != 'human_delibot':
            onboarding_versions.append("|".join(checked_onbording))
            final_versions.append("|".join(checked_final))

        result_stats['onboarding_types'].update([user['ONBOARDING_SUBMIT'][-1][1]])
        result_stats['game_types'].update([user['GAME_SUBMIT'][-1][1]])

        result_stats['message_count'] += len(user['MESSAGES_TOKENIZED'])
        result_stats['tokens_count'] += sum([len(s) for s in user['MESSAGES_TOKENIZED']])
        result_stats['unique_tokens'].update(
            [t.translate(table) for s in user['MESSAGES_TOKENIZED'] for t in s if len(t.strip()) >= 4])

        last = None
        for gs in user['GAME_SUBMIT']:
            if gs[1] != last:
                last = gs[1]
                result_stats['number_of_submits'] += 1

    print(onboarding_versions)
    on_c = Counter(onboarding_versions).most_common(1)[0][1]
    fin_c = Counter(final_versions).most_common(1)[0][1]

    result_stats['num_of_players'] = len(conversations_dump)
    result_stats['num_of_playing_wason'] = len(
        [c for _, c in conversations_dump.items() if c['user_type'] == 'participant'])
    result_stats['onboarding_agreement'] = on_c / len(onboarding_versions)
    result_stats['final_agreement'] = fin_c / len(final_versions)
    result_stats['onboarding_success_rate'] = result_stats['onboarding_score'] / result_stats['num_of_playing_wason']
    result_stats['onboarding_success_rate_coarse'] = result_stats['onboarding_score_coarse'] / result_stats[
        'num_of_playing_wason']

    result_stats['final_success_rate'] = result_stats['game_score'] / result_stats['num_of_playing_wason']
    result_stats['final_success_rate_coarse'] = result_stats['game_score_coarse'] / result_stats['num_of_playing_wason']

    result_stats['message_per_player'] = result_stats['message_count'] / len(conversations_dump)
    result_stats['tokens_per_player'] = result_stats['tokens_count'] / len(conversations_dump)
    result_stats['unique_tokens_count'] = len(result_stats['unique_tokens'])
    result_stats['unique_tokens_per_player'] = len(result_stats['unique_tokens']) / len(conversations_dump)
    result_stats['most_common_tokens'] = result_stats['unique_tokens'].most_common(5)
    result_stats['task_performance'] = result_stats['final_success_rate'] - result_stats['onboarding_success_rate']
    result_stats['performance_gain_binary'] = 1 if result_stats['task_performance'] > 0 else 0
    result_stats['submits_per_user'] = result_stats['number_of_submits'] / result_stats['num_of_playing_wason']


    deli_correct = 0
    if 'CORRECT' not in result_stats['onboarding_types'] and 'CORRECT' in result_stats['game_types']:
        deli_correct = 1

    result_stats['deli_correct'] = deli_correct

    #     del result_stats['unique_tokens']
    return result_stats


def featurise(collection, path):
    stats = []
    for d in collection:
        try:
            prepr = preprocess_conversation_dump(d.raw_db_conversation)
            s = calculate_stats(prepr)


            s['identifier'] = d.identifier
            stats.append(s)

        except Exception as e:
            print(e)

    with open(path, 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for stat in stats:

            csv_writer.writerow([
                                stat['identifier'],
                                stat['num_of_playing_wason'],
                                 stat['message_per_player'],
                                 stat['tokens_per_player'],
                                 stat['unique_tokens_count'],
                                 stat['unique_tokens_per_player'],
                                 stat['onboarding_success_rate'],
                                 stat['onboarding_agreement'],
                                 stat['final_agreement'],
                                stat['message_count']
                                 ])


def get_y(collection):
    Y={}

    for d in collection:
        try:
            prepr = preprocess_conversation_dump(d.raw_db_conversation)
            s = calculate_stats(prepr)

            Y[d.identifier] = s['performance_gain_binary']

        except Exception as e:
            print(e)

    return Y


def get_y_regresion(collection):
    Y={}

    for d in collection:
        try:
            prepr = preprocess_conversation_dump(d.raw_db_conversation)
            s = calculate_stats(prepr)

            Y[d.identifier] = s['task_performance']

        except Exception as e:
            print(e)

    return Y



if __name__ == '__main__':
    # nlp = spacy.load("en_core_web_sm")
    # for a in anns:
    #     a.preprocess_everything(nlp)

    raw_data = read_wason_dump('../data/all_data_20210107/')

    featurise(raw_data, '../features/wason_stats.tsv')
    # stats = []
    # for d in raw_data:
    #     try:
    #         prepr = preprocess_conversation_dump(d.raw_db_conversation)
    #         s = calculate_stats(prepr)
    #         if s['num_of_playing_wason'] >= 2:
    #             stats.append(s)
    #     except Exception as e:
    #         print(e)
    #
    # print()

