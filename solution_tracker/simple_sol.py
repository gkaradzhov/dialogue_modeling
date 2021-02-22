import spacy
import string
from read_data import read_solution_annotaions, read_wason_dump, read_3_lvl_annotation_file
from wason_message import WasonConversation, WasonMessage
import pandas as pd


def solution_tracker(wason_conversation, include_annotations=True, agreement_classifier=None):
    solution_tracker = []
    initial_submissions = {}
    initial_cards = set()

    last_solution = set('0')
    is_solution_proposed_last = False
    last_partial = False
    message_count = 0
    total_length = len(wason_conversation.wason_messages)

    for rm in wason_conversation.raw_db_conversation:
        if rm['message_type'] == "WASON_INITIAL":
            initial_cards.update([l['value'] for l in rm['content']])

        if rm['message_type'] == 'WASON_SUBMIT':
            initial_submissions[rm['user_name']] = set([l['value'] for l in rm['content'] if l['checked']])

        if rm['message_type'] == 'FINISHED_ONBOARDING':
            break

    # Populate initial submissions
    for user, item in initial_submissions.items():
        solution_tracker.append({'type': "INITIAL",
                                 'content': "INITIAL",
                                 'user': user,
                                 'value': item,
                                 'id': -1
                                 })

    # Start tracking conversation

    for item in wason_conversation.raw_db_conversation:
        if item['user_status'] != 'USR_PLAYING':
            continue

        if item['message_type'] == 'WASON_SUBMIT':
            solution_tracker.append({'type': "SUBMIT",
                                     'content': "SUBMIT",
                                     'user': item['user_name'],
                                     'value': set([l['value'] for l in item['content'] if l['checked']]),
                                     'id': item['message_id']
                                     })

        if item['message_type'] == 'CHAT_MESSAGE':
            message_count += 1
            wason_message = wason_conversation.get_wason_from_raw(item)

            if not wason_message:
                continue

            if include_annotations:
                if not wason_message.annotation:
                    continue
                if (wason_message.annotation['target'] in ['Reasoning', 'Disagree', 'Moderation']
                        or wason_message.annotation['type'] == 'Probing') \
                        and len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                            wason_message.annotation['additional'])) == 0:
                    is_solution_proposed_last = False

                cards = {'0'}
                if len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                            wason_message.annotation['additional'])) >= 1:
                    type, cards = extract_from_message(wason_message, initial_cards)

                    if cards != {'0'}:
                        if 'partial_solution' in wason_message.annotation['additional'] and last_solution != {'0'}:
                            if last_partial:
                                last_solution.update(cards)
                                cards = last_solution
                            else:
                                last_partial = True
                                last_solution = cards
                        else:
                            last_solution = cards
                            last_partial = False
                        is_solution_proposed_last = True

                # if cards == {'0'} and wason_message.annotation['target'] == 'Agree' and is_solution_proposed_last:
                #     cards = last_solution

                if len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                        wason_message.annotation['additional'])) >= 1:
                        # or wason_message.annotation['target'] == 'Agree':

                    if cards != {'0'}:
                        solution_tracker.append({'type': "MENTION",
                                                 'content': wason_message.content,
                                                 'user': item['user_name'],
                                                 'value': cards,
                                                 'id': item['message_id'],
                                                 'pos': message_count/total_length
                                                 })
            else:
                type, cards = extract_from_message(wason_message, initial_cards)
                if cards != {'0'}:
                    solution_tracker.append({'type': "MENTION",
                                             'content': wason_message.content,
                                             'user': item['user_name'],
                                             'value': cards,
                                             'id': item['message_id'],
                                             'pos': message_count / total_length
                                             })

                    last_solution = cards

                else:
                    continue
                    agreement = agreement_classifier.predict(wason_message.content)
                    if agreement == 1:
                        solution_tracker.append({'type': "AGREEMENT",
                                                 'content': wason_message.content,
                                                 'user': item['user_name'],
                                                 'value': last_solution,
                                                 'id': item['message_id'],
                                                 'pos': message_count / total_length
                                                 })

    return solution_tracker

def process_raw_to_solution_tracker(wason_conversation: WasonConversation, prediction=False):
    solution_tracker = []
    initial_submissions = {}
    initial_cards = set()
    conf_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    last_solution = set('0')
    is_solution_proposed_last = False
    last_partial = False
    # Get initial submissions
    for rm in wason_conversation.raw_db_conversation:
        if rm['message_type'] == "WASON_INITIAL":
            initial_cards.update([l['value'] for l in rm['content']])

        if rm['message_type'] == 'WASON_SUBMIT':
            initial_submissions[rm['user_name']] = set([l['value'] for l in rm['content'] if l['checked']])

        if rm['message_type'] == 'FINISHED_ONBOARDING':
            break

    # Populate initial submissions
    for user, item in initial_submissions.items():
        solution_tracker.append({'type': "INITIAL",
                                 'content': "INITIAL",
                                 'user': user,
                                 'value': item,
                                 'id': -1
                                 })

    # Start tracking conversation

    for item in wason_conversation.raw_db_conversation:
        if item['user_status'] != 'USR_PLAYING':
            continue

        if item['message_type'] == 'WASON_SUBMIT':
            solution_tracker.append({'type': "SUBMIT",
                                     'content': "SUBMIT",
                                     'user': item['user_name'],
                                     'value': set([l['value'] for l in item['content'] if l['checked']]),
                                     'id': item['message_id']
                                     })

        if item['message_type'] == 'CHAT_MESSAGE':

            wason_message = wason_conversation.get_wason_from_raw(item)

            if not wason_message:
                continue

            if (wason_message.annotation['target'] in ['Reasoning', 'Disagree', 'Moderation']
                    or wason_message.annotation['type'] == 'Probing') \
                    and len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                        wason_message.annotation['additional'])) == 0:
                is_solution_proposed_last = False

            cards = {'0'}
            if len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                        wason_message.annotation['additional'])) >= 1:
                type, cards = extract_from_message(wason_message, initial_cards)

                if cards != {'0'}:
                    if 'partial_solution' in wason_message.annotation['additional'] and last_solution != {'0'}:
                        if last_partial:
                            last_solution.update(cards)
                            cards = last_solution
                        else:
                            last_partial = True
                            last_solution = cards
                    else:
                        last_solution = cards
                        last_partial = False
                    is_solution_proposed_last = True

            # if cards == {'0'} and wason_message.annotation['target'] == 'Agree' and is_solution_proposed_last:
            #     cards = last_solution

            if len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                    wason_message.annotation['additional'])) >= 1:
                    # or wason_message.annotation['target'] == 'Agree':

                if cards != {'0'}:
                    solution_tracker.append({'type': "MENTION",
                                             'content': wason_message.content,
                                             'user': item['user_name'],
                                             'value': cards,
                                             'id': item['message_id']
                                             })

                if not prediction:
                    annotation = wason_message.annotation['sols']
                    if cards == annotation:
                        if cards == {'0'}:
                            conf_matrix['TN'] += 1
                        else:
                            conf_matrix['TP'] += 1
                    else:
                        # print("{} {} {}".format(wason_message.content, cards, annotation))
                        if annotation == {'0'}:
                            conf_matrix['FP'] += 1
                        else:
                            conf_matrix['FN'] += 1

    return conf_matrix, solution_tracker


def extract_from_message(wason_message: WasonMessage, initial_cards, annotations=None):
    mentioned_cards = set()
    for token, pos in zip(wason_message.content_tokenised, wason_message.content_pos):
        stripped = token.translate(str.maketrans('', '', string.punctuation))

        if stripped.upper() in initial_cards:
            mentioned_cards.add(stripped.upper())
        elif stripped == 'all':
            mentioned_cards.update(initial_cards)

    if mentioned_cards:
        return "RAW_MENTION", mentioned_cards
    else:
        return 'NotFound', {'0'}


if __name__ == "__main__":

    anns = read_solution_annotaions('../solution_annotations.tsv')
    nlp = spacy.load("en_core_web_sm")
    for a in anns:
        a.preprocess_everything(nlp)

    raw_data = read_wason_dump('../data/all/')

    conversations_to_process = []
    for ann in anns:
        raw = [r for r in raw_data if r.identifier == ann.identifier][0]
        ann.raw_db_conversation = raw.raw_db_conversation
        conversations_to_process.append(ann)

    hierch_data = read_3_lvl_annotation_file('../3lvl_anns.tsv')

    for conv in anns:
        hierch = [d for d in hierch_data if d.identifier == conv.identifier][0]
        conv.merge_all_annotations(hierch)

    sols = []
    full_conf_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for conv in conversations_to_process:
        local_conf_mat, sol_tracker = process_raw_to_solution_tracker(conv)

        full_conf_matrix['TP'] += local_conf_mat['TP']
        full_conf_matrix['FP'] += local_conf_mat['FP']
        full_conf_matrix['TN'] += local_conf_mat['TN']
        full_conf_matrix['FN'] += local_conf_mat['FN']

        sols.append(sol_tracker)

    accuracy = (full_conf_matrix['TP'] + full_conf_matrix['TN']) /\
               (full_conf_matrix['TP'] + full_conf_matrix['FP'] + full_conf_matrix['TN'] + full_conf_matrix['FN'])
    precision = (full_conf_matrix['TP']) /\
               (full_conf_matrix['TP'] + full_conf_matrix['FP'])
    recall = (full_conf_matrix['TP']) / \
                (full_conf_matrix['TP'] + full_conf_matrix['FN'])
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)

    df = pd.DataFrame(sols[0])

    df.to_csv('sol_tracker_demo.tsv', sep='\t')
    print()



