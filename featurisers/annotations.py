import csv

from sklearn.feature_extraction.text import TfidfVectorizer

from read_data import read_solution_annotaions, read_3_lvl_annotation_file
from utils import SkipGramVectorizer


def init_feature_map():
    return {
        'Probing': 0,
        'Probing_Reasoning': 0,
        'Probing_Solution': 0,
        'Probing_Moderation': 0,
        'Non-probing-deliberation': 0,
        'Non-probing-deliberation_Solution': 0,
        'Non-probing-deliberation_Reasoning': 0,
        'Non-probing-deliberation_Agree': 0,
        'Non-probing-deliberation_Disagree': 0,
        'Reasoning': 0,
        'Solution': 0,
        'Agree': 0,
        'Disagree': 0,
        'Moderation': 0,
        'complete_solution': 0,
        'partial_solution': 0,
        'consider_opposite': 0,
        'solution_summary': 0,
        'specific_addressee': 0,
        '0': 0
    }

def featurise_simple(conversation_collection, path):
    features = []
    for a in conversation_collection:
        f_id = a.identifier
        fmap = init_feature_map()

        for m in a.wason_messages:
            annotation_obj = m.annotation
            if annotation_obj['type'] == '0':
                continue
            fmap[annotation_obj['type']] += 1
            fmap[annotation_obj['target']] += 1
            key = "{}_{}".format(annotation_obj['type'], annotation_obj['target'])
            fmap[key] += 1

            for item in annotation_obj['additional']:
                fmap[item] += 1

        normalised = []
        for key, item in fmap.items():
            normalised.append(float(item / len(a.wason_messages)))
        fmap['has_3_probing'] = 1 if fmap['Probing'] >= 3 else 0
        fmap['has_3_probing_solution'] = 1 if fmap['Probing_Solution'] >= 3 else 0
        fmap['has_3_probing_reasoning'] = 1 if fmap['Probing_Reasoning'] >= 3 else 0

        features.append([f_id, *normalised, fmap['has_3_probing'], fmap['has_3_probing_solution'], fmap['has_3_probing_reasoning']])

    with open(path, 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in features:
            csv_writer.writerow(item)


def featurise_annotation_sequences(conversation_collection, path):
    raw_type = []
    raw_target = []
    raw_both = []
    raw_additional = []
    raw_everything = []

    for a in conversation_collection:
        f_id = a.identifier
        concat_anns_type = []
        concat_anns_target = []
        concat_anns_both = []
        concat_additional = []
        concat_everything = []
        for m in a.wason_messages:
            annotation_obj = m.annotation
            if annotation_obj['type'] == '0':
                continue
            concat_anns_type.append(annotation_obj['type'])
            concat_anns_target.append(annotation_obj['target'])
            concat_anns_both.append("{}_{}".format(annotation_obj['type'], annotation_obj['target']))
            concat_additional.append("*".join(annotation_obj['additional']))
            concat_everything.append("{}_{}_{}".format(annotation_obj['type'], annotation_obj['target'], "*".join(annotation_obj['additional'])))
        raw_type.append([f_id, *concat_anns_type])
        raw_target.append([f_id, *concat_anns_target])
        raw_both.append([f_id, *concat_anns_both])
        raw_additional.append([f_id, *concat_additional])
        raw_everything.append([f_id, *concat_everything])

    tf_idf_type = process_tf_vector(raw_type)
    tf_idf_target = process_tf_vector(raw_target)
    tf_idf_type_target = process_tf_vector(raw_both)
    tf_idf_additional = process_tf_vector(raw_additional)
    tf_idf_everything = process_tf_vector(raw_everything)

    write_file(path + 'tf_idf_type.tsv', tf_idf_type)
    write_file(path + 'tf_idf_target.tsv', tf_idf_target)
    write_file(path + 'tf_idf_type_target.tsv', tf_idf_type_target)
    write_file(path + 'tf_idf_additional.tsv', tf_idf_additional)
    write_file(path + 'tf_idf_everything.tsv', tf_idf_everything)

    sg_type = process_sg_vector(raw_type)
    sg_target = process_sg_vector(raw_target)
    sg_type_target = process_sg_vector(raw_both)
    sg_additional = process_sg_vector(raw_additional)
    sg_everything = process_sg_vector(raw_everything)

    write_file(path + 'sg_type.tsv', sg_type)
    write_file(path + 'sg_target.tsv', sg_target)
    write_file(path + 'sg_type_target.tsv', sg_type_target)
    write_file(path + 'sg_additional.tsv', sg_additional)
    write_file(path + 'sg_everything.tsv', sg_everything)

    write_file('../features/raw_annotations_type.tsv', raw_type)
    write_file('../features/raw_annotations_target.tsv', raw_target)
    write_file('../features/raw_annotations_type_target.tsv', raw_both)
    write_file('../features/raw_annotations_additional.tsv', raw_additional)

def write_file(path, collection):
    with open(path, 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in collection:
            csv_writer.writerow(item)

def identity_tokenizer(text):
    return text

def process_tf_vector(raw):
    only_text = []
    for item in raw:
        # no_id = item[1:]
        # no_id_str = " ".join(no_id[1:])
        only_text.append(item[1:])
    tfIdfvectoriser = TfidfVectorizer(ngram_range=(1, 5), tokenizer=identity_tokenizer, preprocessor=None,
                                  lowercase=False, max_features=500)
    tfIdfvectoriser.fit(only_text)
    processed = []
    for doc in raw:
        trans = tfIdfvectoriser.transform([doc[1:]])[0].toarray()[0]
        processed.append([doc[0], *trans])

    return processed

def process_sg_vector(raw):
    only_text = []
    for item in raw:
        # no_id = item[1:]
        # no_id_str = " ".join(no_id[1:])
        only_text.append(item[1:])
    tfIdfvectoriser = SkipGramVectorizer(ngram_range=(1, 5), tokenizer=identity_tokenizer, preprocessor=None,
                                  lowercase=False, max_features=500)
    tfIdfvectoriser.fit(only_text)
    processed = []
    for doc in raw:
        trans = tfIdfvectoriser.transform([doc[1:]])[0].toarray()[0]
        processed.append([doc[0], *trans])

    return processed


if __name__ == "__main__":

    anns = read_3_lvl_annotation_file('../3lvl_anns.tsv')
    featurise_simple(anns, '../features/annotation.tsv')
    featurise_annotation_sequences(anns, '../features/annotations_')


    pass