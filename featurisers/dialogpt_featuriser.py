import csv

import spacy
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


from read_data import read_wason_dump

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    raw_data = read_wason_dump('../data/all_data_20210107/')
    for item in raw_data:
        item.wason_messages_from_raw()
        item.preprocess_everything(nlp)
        item.remove_solutions()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    lm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", output_hidden_states=True)

    for conversation in raw_data:
        chat_history_ids = None

        for item in conversation.wason_messages:
            user_input = tokenizer.encode(item.no_solution_text + tokenizer.eos_token, return_tensors='pt')
            chat_history_ids = torch.cat([chat_history_ids, user_input],
                                      dim=-1) if chat_history_ids is not None else user_input
            # chat_history_ids = lm_model.generate(bot_input_ids, max_length=3000, pad_token_id=tokenizer.eos_token_id)
            print(chat_history_ids.shape)
        print('=============')
        lm_out = lm_model(chat_history_ids)
        with open('../features/dialogpt_pretrained/' + conversation.identifier + '.tsv', 'w') as wf:
            tsv_writer = csv.writer(wf, delimiter='\t')
            interested_row = lm_out[2][-1]
            for item in interested_row[0]:
                tsv_writer.writerow(item.detach().numpy())