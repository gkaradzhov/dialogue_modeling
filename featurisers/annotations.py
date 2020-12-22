import csv

from sklearn.feature_extraction.text import TfidfVectorizer

from read_data import read_solution_annotaions, read_3_lvl_annotation_file


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


def featurise_tf_idfs(conversation_collection, path):
    raw_type = []
    raw_target = []
    raw_both = []

    for a in conversation_collection:
        f_id = a.identifier
        concat_anns_type = []
        concat_anns_target = []
        concat_anns_both = []
        for m in a.wason_messages:
            annotation_obj = m.annotation
            if annotation_obj['type'] == '0':
                continue
            concat_anns_type.append(annotation_obj['type'])
            concat_anns_target.append(annotation_obj['target'])
            concat_anns_both.append("{}_{}".format(annotation_obj['type'], annotation_obj['target']))

        raw_type.append([f_id, *concat_anns_type])
        raw_target.append([f_id, *concat_anns_target])
        raw_both.append([f_id, *concat_anns_both])

    type_features = process_tf_vector(raw_type)
    target_features = process_tf_vector(raw_target)
    both_features = process_tf_vector(raw_both)

    with open(path + 'type.tsv', 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in type_features:
            csv_writer.writerow(item)

    with open(path + 'target.tsv', 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in target_features:
            csv_writer.writerow(item)

    with open(path + 'both.tsv', 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in both_features:
            csv_writer.writerow(item)


def identity_tokenizer(text):
    return text

def process_tf_vector(raw):
    only_text = []
    for item in raw:
        # no_id = item[1:]
        # no_id_str = " ".join(no_id[1:])
        only_text.append(item[1:])
    tfIdfvectoriser = TfidfVectorizer(ngram_range=(2, 5), tokenizer=identity_tokenizer, preprocessor=None, lowercase=False, max_features=200)
    tfIdfvectoriser.fit(only_text)
    processed = []
    for doc in raw:
        trans = tfIdfvectoriser.transform([doc[1:]])[0].toarray()[0]
        processed.append([doc[0], *trans])

    return processed


if __name__ == "__main__":

    anns = read_3_lvl_annotation_file('../3lvl_anns.tsv')
    featurise_simple(anns, '../features/annotation.tsv')
    featurise_tf_idfs(anns, '../features/annotations_tf_idf_2_5')


    pass