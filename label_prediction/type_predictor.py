from itertools import combinations
from pprint import pprint

import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from scipy import stats

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModelForCausalLM

from prediction_utils import get_features, merge_feauters, features_labels_to_xy, \
    decision_tree_representation, decision_tree_stats, logging
from read_data import read_wason_dump, read_3_lvl_annotation_file
import numpy as np
from solution_tracker.augment_with_solution import merge_with_solution_annotation_message_level


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):

        return [d[self.key] for d in data_dict]


class DialoGPT(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.lm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", output_hidden_states=True)


    def fit(self,  x, y=None):
        return self

    def transform(self, x):
        result = []

        for item in x:

            user_input = self.tokenizer.encode(item + self.tokenizer.eos_token, return_tensors='pt')
            lm_out = self.lm_model(user_input)
            output = lm_out[2][-1].detach().numpy().mean(axis=1)[0]
            result.append(output)
                # interested_row = lm_out[2][-1]
                # for item in interested_row[0]:
                #     tsv_writer.writerow(item.detach().numpy())
        return result

from wason_message import WasonMessage

if __name__ == '__main__':
    # 1. Read Labels
    raw_data = read_wason_dump('../data/all_data_20210107/')

    hierch_data = read_3_lvl_annotation_file('../3lvl_anns.tsv')
    nlp = spacy.load("en_core_web_sm")
    for a in hierch_data:
        a.preprocess_everything(nlp)

    conversations_to_process = []
    for conv in hierch_data:
        raw = [d for d in raw_data if d.identifier == conv.identifier][0]
        conv.raw_db_conversation = raw.raw_db_conversation
        # messages_unsupervised = merge_with_solution_annotation_message_level(conv, False)
        # conv.wason_messages = messages_unsupervised
        conversations_to_process.append(conv)


    X_raw, Y, X_target = [], [], []
    mode = "type"
    m = WasonMessage(origin='SYS', content='BEGIN', identifier='-12', annotation_obj={})
    for conversation in conversations_to_process:
        for prev, message in zip([m, *conversation.wason_messages[:-1]], conversation.wason_messages):
            if message.origin == "Moderating Owl":
                print("owl")
                continue
            X_raw.append({'context': prev.content, 'current': message.content})
            X_target.append(message.annotation['target'])
            if mode == 'additional_label':
                if 'partial_solution' in message.annotation['additional']:
                    label = 'partial_solution'
                elif 'complete_solution' in message.annotation['additional']:
                    label = 'complete_solution'
                else:
                    label = '0'
            else:
                label = message.annotation['type']
            Y.append(label)


    print(set(Y))

    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the post's subject line
                ('context_tf', Pipeline([
                    ('selector', ItemSelector(key='context')),
                    ('tfidf_1', TfidfVectorizer(ngram_range=(1, 5), max_features=10000)),
                ])),
                ('current_tf', Pipeline([
                    ('selector', ItemSelector(key='current')),
                    ('tfidf_2', TfidfVectorizer(ngram_range=(1, 5), max_features=10000)),
                ])),

                # ('context_gpt', Pipeline([
                #     ('selector', ItemSelector(key='context')),
                #     ('gpt1', DialoGPT()),
                # ])),
                ('current_gpt', Pipeline([
                    ('selector', ItemSelector(key='current')),
                    ('gpt2', DialoGPT()),
                ])),
            ],
            transformer_weights={
                'context_tf': 0,
                'current_tf': 1,
                # 'context_gpt': 1,
                # 'current_gpt': 1,
            },
        )),
        ('scale', MaxAbsScaler()),
        ('clf', SGDClassifier()),
    ])
    parameters = [

        {
            'union__context_tf__tfidf_1__ngram_range': ((1, 2), (1, 3),(1, 4)),
            'union__current_tf__tfidf_2__ngram_range': ((1, 2), (1, 3),(1, 4)),
            'clf': (SVC(kernel='rbf'),),
            'clf__C': (0.1, 0.03, 0.01),
            'clf__gamma': (0.001, 0.01, 0.1),
            'clf__random_state': (42,),
        },
        {
            'union__context_tf__tfidf_1__ngram_range': ((1, 2), (1, 3),(1, 4)),
            'union__current_tf__tfidf_2__ngram_range': ((1, 2), (1, 3),(1, 4)),
            'clf': (RandomForestClassifier(),),
            'clf__n_estimators': (10, 50),
            'clf__random_state': (42,),
            'clf__class_weight': ('balanced', None),
        },
        {

            'clf': (DecisionTreeClassifier(random_state=42),),
            'clf__max_depth': (5,),
            'clf__min_samples_leaf': (5,)
        },
        {
            'clf': (KNeighborsClassifier(),),
            'clf__n_neighbors': (3, 5, 7, 9),
            'clf__metric': ('euclidean', 'minkowski'),
        },

    ]
    grid_search = GridSearchCV(pipeline, parameters, cv=8, n_jobs=4, scoring='f1_macro', verbose=3)

    clf = grid_search.fit(X_raw, Y)

    best_estimator = clf.best_estimator_

    # 6. LOOCV
    loo = KFold(n_splits=10)

    predicted = []
    gold = []
    targets = []
    texts = []

    X_raw = np.array(X_raw)
    Y = np.array(Y)
    X_target_np = np.array(X_target)

    for train_index, test_index in loo.split(X_raw):
        print("new split")
        X_train, X_test = X_raw[train_index], X_raw[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        x_target_train, x_target_test = X_target_np[train_index], X_target_np[test_index]
        new_fit = best_estimator.fit(X_train, y_train)
        print(set(y_train))
        pred = new_fit.predict(X_test)
        # print("{} ::: {}".format(new_fit.predict_proba(X_test), y_test))
        predicted.extend(pred)
        gold.extend(y_test)
        targets.extend(x_target_test)
        texts.extend(X_test)
        # dt_repres.append(decision_tree_representation(new_fit['clf']))

    clas_rep = classification_report(gold, predicted)
    performance = f1_score(gold, predicted, average='macro')
    # dt_stats = decision_tree_stats(dt_repres)
    print(clas_rep)
    print(clf.best_params_)

    logging('predictor_naive_{}.tsv'.format(mode), "n/a", clf.best_params_, performance)

    targets_predictions = {
        'Probing':
            {
                'Reasoning': [0, 0],
                'Solution': [0, 0],
                'Moderation': [0, 0],
                'Agree': [0, 0],
                'Disagree': [0, 0]
            },
        'Non-probing-deliberation':
            {
                'Reasoning': [0, 0],
                'Solution': [0, 0],
                'Moderation': [0, 0],
                'Agree': [0, 0],
                'Disagree': [0, 0],
            },
        '0': {'0': [0, 0]}

    }

    errors = []
    for g, p, t, raw in zip(gold, predicted, targets, texts):
        if g != p:
            targets_predictions[g][t][1] += 1
            errors.append([g, p, t, raw])
        else:
            targets_predictions[g][t][0] += 1

    print("Gold\tPredicted\tTarget\tInput")
    for g, p, t, raw in errors:
        print("{}\t{}\t{}\t{}".format(g, p, t, raw['current']))

    pprint(targets_predictions, indent=4)



