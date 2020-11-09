import pandas as pd
import json
from collections import Counter
import ast

import os

import spacy

from wason_message import WasonConversation, WasonMessage


def read_wason_dump(dump_path):
    wason_dump = pd.DataFrame()
    return wason_dump


def read_3_lvl_annotation_file(ann_path):
    annotations_df = pd.read_csv(ann_path, delimiter='\t')
    annotations_df = annotations_df.fillna('0')

    processed_dialogue_annotations = {}
    current_dialogue = []
    for item in annotations_df.iterrows():
        if item[1]['room_id'] == '0' and len(current_dialogue) > 0:
            room_id = current_dialogue[-1]['room_id']
            processed_dialogue_annotations[room_id] = current_dialogue
            current_dialogue = []
        current_dialogue.append({
                                'origin': item[1]['Origin'],
                                'room_id': item[1]['room_id'], 'content': item[1]['Content'], 'type': item[1]['Type'],
                                 'role': item[1]['Target'],
                                 'labels': [b.strip() for b in item[1]['Additional'].split(',')]})

    room_id = current_dialogue[-1]['room_id']
    processed_dialogue_annotations[room_id] = current_dialogue

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
                               annotation_obj=set([b.strip() for b in item[1]['Solutions'].split(',')]))
        current_dialogue.append(message)

    wason_conversation = WasonConversation(last_room_id)
    wason_conversation.wason_messages = current_dialogue
    processed_dialogue_annotations.append(wason_conversation)

    return processed_dialogue_annotations


if __name__ == '__main__':
    anns = read_solution_annotaions('solution_annotations.tsv')
    nlp = spacy.load("en_core_web_sm")
    for a in anns:
        a.pos_tag_everything(nlp)

    print()
