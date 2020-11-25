import spacy
import string
from read_data import read_solution_annotaions, read_wason_dump, read_3_lvl_annotation_file
from wason_message import WasonConversation, WasonMessage
import pandas as pd

def process_raw_to_solution_tracker(wason_conversation: WasonConversation):
    solution_tracker = []
    initial_submissions = {}
    initial_cards = set()
    total_anns = 0
    correct_anns = 0
    prec_total = 0
    prec_correct = 0

    last_solution = set('0')
    is_solution_proposed_last = False
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
        # if item['message_type'] == 'WASON_GAME':
        #     solution_tracker.append({'type': "CLICK",
        #                              'content': "CLICK",
        #                              'user': item['user_name'],
        #                              'value': set([l['value'] for l in item['content'] if l['checked']]),
        #                              'id': item['message_id']
        #                              })

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
                    last_solution = cards
                    is_solution_proposed_last = True

            if cards == {'0'} and wason_message.annotation['target'] == 'Agree' and is_solution_proposed_last:
                cards = last_solution

            if len({'partial_solution', 'complete_solution', 'solution_summary'}.intersection(
                    wason_message.annotation['additional'])) >= 1 or wason_message.annotation['target'] == 'Agree':
                total_anns += 1

                if wason_message.annotation['sols'] != {'0'}:
                    prec_total += 1

                if cards == wason_message.annotation['sols']:
                    correct_anns += 1
                    if wason_message.annotation['sols'] != {'0'}:
                        prec_correct += 1
                else:
                    pass
                    # print("{} : {} : {}".format(wason_message.content, wason_message.annotation['sols'], cards))

                if cards != {'0'}:
                    solution_tracker.append({'type': "MENTION",
                                             'content': wason_message.content,
                                             'user': item['user_name'],
                                             'value': cards,
                                             'id': item['message_id']
                                             })

    return correct_anns, total_anns, prec_correct, prec_total, solution_tracker
    pass


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

    correct = 0
    total = 0
    p_correct = 0
    p_total = 0

    sols = []
    for conv in conversations_to_process:
        c_correct, c_total, cp_correct, cp_total, sol_tracker = process_raw_to_solution_tracker(conv)
        correct += c_correct
        total += c_total
        p_correct += cp_correct
        p_total += cp_total
        sols.append(sol_tracker)

    print(correct/total)
    print(p_correct/p_total)

    df = pd.DataFrame(sols[0])

    df.to_csv('sol_tracker_demo.tsv', sep='\t')
    print()



