from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scipy import stats

from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold

from featurisers.raw_wason_featuriser import get_y
from prediction_utils import get_features, merge_feauters, features_labels_to_xy, \
    decision_tree_representation, decision_tree_stats, logging
from read_data import read_wason_dump
import numpy as np

FEATURE_MAPS = {
    'street_crowd_turns': '../features/sc_turns.tsv',
    'street_crowd_messages': '../features/sc_messages.tsv',
    'annotation_features': '../features/annotation.tsv',
    'dialogue_metadata': '../features/wason_stats.tsv',
    'text_features': '/',
    'positive_correlations': '../features/positive_correlation_features.tsv',
    'solution_participation_automatic': '../features/solution_participation_automatic.tsv',
    'solution_participation': '../features/solution_participation.tsv',
    'annotation_tfidf_type': '../features/annotations_tf_idf_type.tsv',
    'annotation_tfidf_target': '../features/annotations_tf_idf_target.tsv',
    'annotation_tfidf_type_target': '../features/annotations_tf_idf_type_target.tsv',
    'annotation_tfidf_additional': '../features/annotations_tf_idf_additional.tsv',
    'annotation_tfidf_everything': '../features/annotations_tf_idf_everything.tsv',
    'annotation_sg_type': '../features/annotations_sg_type.tsv',
    'annotation_sg_target': '../features/annotations_sg_target.tsv',
    'annotation_sg_type_target': '../features/annotations_sg_type_target.tsv',
    'annotation_sg_additional': '../features/annotations_sg_additional.tsv',
    'annotation_sg_everything': '../features/annotations_sg_everything.tsv',
    'nn_representation': '../features/nn_representation.tsv',
    'random_vector': '../features/random_vector.tsv',
    'fast_text': '../features/fast_text_representation_20.tsv',
    'fast_text_additional': '../features/fast_text_representation_additional_20.tsv',
    'fast_text_uni': '../features/fast_text_representation_20_uni.tsv',
    'fast_text_additional_uni': '../features/fast_text_representation_additional_20_uni.tsv',
    'fast_text_sc': '../features/fast_text_representation_sc.tsv',
    'fast_text_stats': '../features/fast_text_representation_stats.tsv',
    'ids_2': '../features/ids_2',
    'ids_3': '../features/ids_3',
    'ids_4_5': '../features/ids_4_5',
    'ids_multi': '../features/ids_milti',
    'ids_all': '../features/ids_all'
}

if __name__ == '__main__':
    # 1. Read Labels
    raw_data = read_wason_dump('../data/final_all/')
    Y_raw = get_y(raw_data)

    ids = get_features(FEATURE_MAPS, 'ids_all')
    print(len(ids))

    # 2. Get features
    meta_feats = get_features(FEATURE_MAPS, 'dialogue_metadata')
    annotation = get_features(FEATURE_MAPS, 'annotation_features')

    sc_turns = get_features(FEATURE_MAPS, 'street_crowd_turns')

    sc_messages = get_features(FEATURE_MAPS, 'street_crowd_messages')
    sol_part = get_features(FEATURE_MAPS, 'solution_participation')
    pos_cor = get_features(FEATURE_MAPS, 'positive_correlations')

    tf_type = get_features(FEATURE_MAPS, 'annotation_tfidf_type')
    tf_target = get_features(FEATURE_MAPS, 'annotation_tfidf_target')
    tf_type_target = get_features(FEATURE_MAPS, 'annotation_tfidf_type_target')
    tf_additional = get_features(FEATURE_MAPS, 'annotation_tfidf_additional')
    tf_everything = get_features(FEATURE_MAPS, 'annotation_tfidf_everything')


    sg_type = get_features(FEATURE_MAPS, 'annotation_sg_type')
    sg_target = get_features(FEATURE_MAPS, 'annotation_sg_target')
    sg_type_target = get_features(FEATURE_MAPS, 'annotation_sg_type_target')
    sg_additional = get_features(FEATURE_MAPS, 'annotation_sg_additional')
    sg_everything = get_features(FEATURE_MAPS, 'annotation_sg_everything')

    nn_representation = get_features(FEATURE_MAPS, 'nn_representation')
    random_vector = get_features(FEATURE_MAPS, 'random_vector')
    fast_text_ngram = get_features(FEATURE_MAPS, 'fast_text')
    fast_text_additional_ngram = get_features(FEATURE_MAPS, 'fast_text_additional')
    fast_text_uni = get_features(FEATURE_MAPS, 'fast_text_uni')
    fast_text_additional_uni = get_features(FEATURE_MAPS, 'fast_text_additional_uni')
    fast_text_sc = get_features(FEATURE_MAPS, 'fast_text_sc')
    fast_text_stats = get_features(FEATURE_MAPS, 'fast_text_stats')

    feature_combinations = {
        'meta_feats': meta_feats,
        # 'annotation': annotation,
        'sc_turns': sc_turns,
        'sc_messages': sc_messages,
        'solution_participation': sol_part,
        # 'annotation_tf_type': tf_type,
        # 'annotation_tf_target': tf_target,
        # 'annotation_tf_type_target': tf_type_target,
        # 'annotation_tf_additional': tf_additional,
        # 'annotation_tf_everything': tf_everything,

        # 'annotation_sg_type': sg_type,
        # 'annotation_sg_target': sg_target,
        # 'annotation_sg_type_target': sg_type_target,
        # 'annotation_sg_additional': sg_additional,
        # 'annotation_sg_everything': sg_everything,
        # 'nn_representation': nn_representation,
        # 'random_vector': random_vector,
        # 'fast_text_20_uni': fast_text_uni,
        # 'fast_text_additional_20_uni': fast_text_additional_uni,
        # 'fast_text_20_ngram': fast_text_ngram,
        # 'fast_text_additional_20_ngram': fast_text_additional_ngram,
        # 'fast_text_SC': fast_text_sc,
        # 'fast_text_stats': fast_text_stats
        # 'pos_cor': pos_cor
    }
    combs = []

    for i in range(1, len(feature_combinations) + 1):
        els = [list(x) for x in combinations(feature_combinations.keys(), i)]
        combs.extend(els)

    # combs = [[f] for f in feature_combinations.keys()]
    # combs = [['sc_messages', 'sc_turns'], ['annotation', 'sc_turns']]
    counter = 0
    for comb in combs:
        print("Combination: {}/{} ".format(counter, len(combs)))
        counter += 1
        merged_feats = merge_feauters([feature_combinations[c] for c in comb])

        X, Y = features_labels_to_xy(merged_feats, Y_raw, ids)
        X = np.array(X)
        print("X Len: ", len(X))
        # 4. Create pipeline

        pipeline = Pipeline([
            ('scale', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', SGDClassifier()),
        ])
        parameters = [

            # {
            #     'clf': (SVC(probability=True),),
            #     'clf__C': (0.01, 1.0, 2),
            #     'clf__kernel': ('rbf', 'linear', 'poly'),
            #     'clf__gamma': (0.001, 0.01),
            #     'clf__random_state': (42,),
            # },
            # {
            #     'clf': (RandomForestClassifier(),),
            #     'clf__n_estimators': (3, 5, 10),
            #     'clf__random_state': (42,),
            #     'clf__criterion': ("gini", "entropy"),
            #     'clf__class_weight': ('balanced', None),
            # },
            # {'clf': (LogisticRegression(),)},
            {
                'clf': (DecisionTreeClassifier(random_state=42),),
                'clf__max_depth': (7,),
                'clf__min_samples_leaf': (5,)
            },
            # {
            #     'clf': (KNeighborsClassifier(),),
            #     'clf__n_neighbors': (3, 5, 7, 9),
            #     'clf__metric': ('euclidean', 'minkowski'),
            # },

        ]
        grid_search = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1)

        clf = grid_search.fit(X, Y)

        best_estimator = clf.best_estimator_

        # 6. LOOCV
        # loo = KFold(n_splits=10)
        loo = LeaveOneOut()
        predicted = []
        gold = []
        dt_repres = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            new_fit = best_estimator.fit(X_train, y_train)

            pred = new_fit.predict(X_test)
            # print("{} ::: {}".format(new_fit.predict_proba(X_test), y_test))
            predicted.extend(pred)
            gold.extend(y_test)

            dt_repres.append(decision_tree_representation(new_fit['clf']))

        clas_rep = classification_report(gold, predicted, output_dict=True)

        dt_stats = decision_tree_stats(dt_repres)
        print(clas_rep)
        print(clf.best_params_)
        # print(clf.best_score_)
        performance = roc_auc_score(gold, predicted)
        print(performance)

        # print(X.shape)
        mode = stats.mode(Y)
        occs = np.count_nonzero(Y == mode)
        baseline = roc_auc_score(gold, [mode[0]]*len(Y))
        print('Baseline: {}'.format(baseline))
        print(comb)
        res = []

        for p, g in zip(predicted, gold):
            if p == g:
                res.append(1)
            else:
                res.append(0)

        print(res)
        print('=================')
        logging('classification_allids.tsv', comb, clf.best_params_, performance, dt_stats)

