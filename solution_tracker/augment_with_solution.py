import copy

import spacy

from featurisers.raw_wason_featuriser import is_solution_fine_grained
from read_data import read_3_lvl_annotation_file, read_wason_dump
from solution_tracker.simple_sol import solution_tracker
from supporting_classifiers.agreement_classifier import Predictor
from wason_message import WasonMessage


def calculate_team_performance(latest_solutions):
    score = 0
    for u, solution in latest_solutions.items():
        local_score, _ = is_solution_fine_grained(solution)
        score += local_score

    return round(score / len(latest_solutions), 3)


def merge_with_solution(conversation_external, supervised=True):
    conversation = copy.deepcopy(conversation_external)
    if not supervised:
        agreement_predictor = Predictor('models/agreement.pkl')
        sol_tracker = solution_tracker(conversation, supervised, agreement_predictor)
    else:
        sol_tracker = solution_tracker(conversation, supervised)


    with_solutions = []

    latest_sol = {}

    for item in sol_tracker:
        if item['type'] == 'INITIAL':
            latest_sol[item['user']] = " ".join(item['value'])
        else:
            if item['user'] not in latest_sol:
                latest_sol[item['user']] = 'N/A'

    team_performance = calculate_team_performance(latest_sol)

    latest_score = team_performance

    for raw in conversation.raw_db_conversation:
        if raw['user_status'] != 'USR_PLAYING':
            continue

        if raw['message_type'] == 'WASON_SUBMIT':
            local_sol = [s for s in sol_tracker if s['id'] == raw['message_id']][0]
        elif raw['message_type'] == 'CHAT_MESSAGE':
            local_sol = [s for s in sol_tracker if s['id'] == raw['message_id']]
            if len(local_sol) == 0:
                if raw['user_name'] not in latest_sol:
                    latest_sol[raw['user_name']] = 'UKN'
                local_sol = {'user': raw['user_name'], 'value': latest_sol[raw['user_name']]}
            else:
                local_sol = local_sol[0]
        else:
            continue

        latest_sol[local_sol['user']] = local_sol['value']
        team_performance = calculate_team_performance(latest_sol)
        display_dict = copy.copy(latest_sol)
        display_dict['team_performance'] = team_performance
        display_dict['performance_change'] = team_performance - latest_score
        latest_score = team_performance

        annotation_wason_conv = None

        for index, item in enumerate(conversation.wason_messages):
            if item.identifier == raw['message_id']:
                annotation_wason_conv = item
                last_index = index
        if annotation_wason_conv is not None:
            annotation_wason_conv.annotation.update(display_dict)
            with_solutions.append(annotation_wason_conv)
        else:
            wm = WasonMessage(identifier=raw['message_id'], origin=raw['user_name'], content='SYSTEM',
                              annotation_obj=display_dict)
            with_solutions.append(wm)

    return with_solutions


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    hierch_data = read_3_lvl_annotation_file('../3lvl_anns.tsv')

    for a in hierch_data:
        a.preprocess_everything(nlp)

    raw_data = read_wason_dump('../data/all/')

    conversations_to_process = {}
    for ann in hierch_data:
        raw = [r for r in raw_data if r.identifier == ann.identifier][0]
        ann.raw_db_conversation = raw.raw_db_conversation
        conversations_to_process[ann.identifier] = ann

    for conv in conversations_to_process.values():
        res = merge_with_solution(conv)

        print(res)
