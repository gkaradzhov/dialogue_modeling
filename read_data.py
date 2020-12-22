import traceback

import pandas as pd
import json
from collections import Counter
import ast
import csv
import os

import spacy

from wason_message import WasonConversation, WasonMessage


def read_wason_dump(dump_path):
    files = os.listdir(dump_path)
    conversations = []
    for f in files:
        conv = WasonConversation(identifier=f.split('.')[0])

        try:
            with open(dump_path + f, 'r') as rf:
                csv_reader = csv.reader(rf, delimiter='\t')
                next(csv_reader)  # Skip header row
                for item in csv_reader:
                    if item[3] in ['WASON_INITIAL', 'WASON_GAME', 'WASON_SUBMIT']:
                        item[4] = item[4].replace('false', 'False').replace('true', 'True')

                    content = item[4]
                    if item[3] in ['WASON_INITIAL', 'WASON_GAME', 'WASON_SUBMIT']:
                        content = ast.literal_eval(item[4])
                    else:
                        try:
                            content = ast.literal_eval(item[4])['message']
                        except Exception as e:
                            pass
                    if len(item) < 7:
                        conv.raw_db_conversation.append({
                            'message_id': item[0],
                            'user_name': item[1],
                            'user_id': item[2],
                            'message_type': item[3],
                            'content': content,
                            'user_status': item[5],
                            'timestamp': item[6],
                            'user_type': 'participant'

                        })
                    else:
                        conv.raw_db_conversation.append({
                            'message_id': item[0],
                            'user_name': item[1],
                            'user_id': item[2],
                            'message_type': item[3],
                            'content': content,
                            'user_status': item[5],
                            'timestamp': item[6],
                            'user_type': item[7] if len(item[7]) > 0 else 'participant'
                        })

        except Exception as e:
            traceback.print_exc()
            print(e)
            print(f)

        conversations.append(conv)

    return conversations


def read_3_lvl_annotation_file(ann_path):
    solutions_df = pd.read_csv(ann_path, delimiter='\t')
    solutions_df = solutions_df.fillna('0')

    processed_dialogue_annotations = []
    current_dialogue = []
    last_room_id = '0'
    for item in solutions_df.iterrows():
        if item[1]['room_id'] == '0' and len(current_dialogue) > 0 and last_room_id != '0':
            wason_conversation = WasonConversation(last_room_id)
            wason_conversation.wason_messages = current_dialogue
            processed_dialogue_annotations.append(wason_conversation)
            current_dialogue = []
        last_room_id = item[1]['room_id']
        message = WasonMessage(origin=item[1]['Origin'], content=item[1]['Content'],
                               annotation_obj=
                               {
                                   'additional': set([b.strip() for b in item[1]['Additional'].split(',')]),
                                   'type': item[1]['Type'],
                                   'target': item[1]['Target']
                                },
                               identifier=item[1]['Message_id'])
        current_dialogue.append(message)

    wason_conversation = WasonConversation(last_room_id)
    wason_conversation.wason_messages = current_dialogue
    processed_dialogue_annotations.append(wason_conversation)

    return processed_dialogue_annotations


def read_solution_annotaions(sol_path):

    solutions_df = pd.read_csv(sol_path, delimiter='\t')
    solutions_df = solutions_df.fillna('0')

    processed_dialogue_annotations = []
    current_dialogue = []
    last_room_id = '0'
    for item in solutions_df.iterrows():
        if item[1]['room_id'] == '0' and len(current_dialogue) > 0 and last_room_id != '0':
            wason_conversation = WasonConversation(last_room_id)
            wason_conversation.wason_messages = current_dialogue
            processed_dialogue_annotations.append(wason_conversation)
            current_dialogue = []
        last_room_id = item[1]['room_id']
        message = WasonMessage(origin=item[1]['Origin'], content=item[1]['Content'],
                               annotation_obj={'sols': set([b.strip() for b in item[1]['Solutions'].split(',')])},
                               identifier=item[1]['Message_id'])
        current_dialogue.append(message)

    wason_conversation = WasonConversation(last_room_id)
    wason_conversation.wason_messages = current_dialogue
    processed_dialogue_annotations.append(wason_conversation)

    return processed_dialogue_annotations


if __name__ == '__main__':
    anns = read_solution_annotaions('solution_annotations.tsv')
    # nlp = spacy.load("en_core_web_sm")
    # for a in anns:
    #     a.preprocess_everything(nlp)

    raw_data = read_wason_dump('data/all/')

    hierch_data = read_3_lvl_annotation_file('3lvl_anns.tsv')

    conversations_to_process = []
    for ann in anns:
        raw = [r for r in raw_data if r.identifier == ann.identifier][0]
        ann.raw_db_conversation = raw.raw_db_conversation
        conversations_to_process.append(ann)


    for conv in anns:
        hierch = [d for d in hierch_data if d.identifier == conv.identifier][0]
        conv.merge_all_annotations(hierch)


    print()
