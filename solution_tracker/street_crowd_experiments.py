import spacy

from external_tools.cornellversation.constructive.msg_features import message_features
from read_data import read_solution_annotaions

if __name__ == '__main__':
    anns = read_solution_annotaions('../solution_annotations.tsv')
    nlp = spacy.load("en_core_web_sm")
    for a in anns:
        a.pos_tag_everything(nlp)

    sc_format = []
    for conv in anns:
        sc_format.append(conv.to_street_crowd_format())

    test = message_features(sc_format[0])
    for utt, t in zip(sc_format[0], test[0]):
        print(" {} @@ {}".format(utt[1], t))