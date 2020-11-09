import spacy

from external_tools.cornellversation.constructive.msg_features import message_features
from read_data import read_solution_annotaions

if __name__ == '__main__':
    anns = read_solution_annotaions('../solution_annotations.tsv')
    nlp = spacy.load("en_core_web_sm")
    for a in anns:
        a.preprocess_everything(nlp)

    sc_format = []
    for conv in anns:
        sc_format.append(conv.to_street_crowd_format())


    for conversation in sc_format:
        test = message_features(conversation)

        print("Introduced Ideas:")
        for intro_ideas in test[3]:
            print(intro_ideas)

        print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
        
        
        print("Repeated Ideas:")
        for repeated_ideas in test[2]:
            print(repeated_ideas)
            
        print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
        print("Conversation")
        for item in conversation:
            print(item)
            
        print('====================================================================================================')


