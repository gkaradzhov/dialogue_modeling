from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from itertools import combinations

from featurisers.raw_wason_featuriser import get_y_regresion
from outcome_prediction.prediction_utils import get_features, logging, merge_feauters, features_labels_to_xy
from read_data import read_wason_dump

import numpy as np

FEATURE_MAPS = {
    'street_crowd_turns': '../features/sc_turns.tsv',
    'street_crowd_messages': '../features/sc_messages.tsv',
    'annotation_features': '../features/annotation.tsv',
    'dialogue_metadata': '../features/wason_stats.tsv',
    'text_features': '/',
    'solution_participation': '../features/solution_participation_automatic.tsv',
    'annotation_tfidf_type': '../features/annotations_tf_idf_2_5type.tsv',
    'annotation_tfidf_target': '../features/annotations_tf_idf_2_5target.tsv',
    'annotation_tfidf_both': '../features/annotations_tf_idf_2_5both.tsv',
    'positive_correlations': '../features/positive_correlation_features.tsv'
}



if __name__ == '__main__':
    # 1. Read Labels
    raw_data = read_wason_dump('../data/all_data_20210107/')
    Y_raw = get_y_regresion(raw_data)

    # 2. Get features
    meta_feats = get_features(FEATURE_MAPS, 'dialogue_metadata')
    annotation = get_features(FEATURE_MAPS, 'annotation_features')
    sc_turns = get_features(FEATURE_MAPS, 'street_crowd_turns')
    sc_messages = get_features(FEATURE_MAPS, 'street_crowd_messages')
    sol_part = get_features(FEATURE_MAPS, 'solution_participation')
    tf_type = get_features(FEATURE_MAPS, 'annotation_tfidf_type')
    tf_target = get_features(FEATURE_MAPS, 'annotation_tfidf_target')
    tf_both = get_features(FEATURE_MAPS, 'annotation_tfidf_both')
    pos_cor = get_features(FEATURE_MAPS, 'positive_correlations')

    feature_combinations = {
        'meta_feats': meta_feats,
        # 'annotation': annotation,
        'sc_turns': sc_turns,
        'sc_messages': sc_messages,
        'solution_participation': sol_part,
        # # 'annotation_tf_type': tf_type,
        # # 'annotation_tf_target': tf_target,
        # 'annotation_tf_both': tf_both,
        # 'pos_cor': pos_cor
    }
    combs = []

    for i in range(1, len(feature_combinations) + 1):
        els = [list(x) for x in combinations(feature_combinations.keys(), i)]
        combs.extend(els)

    for comb in combs:
        print(comb)
        merged_feats = merge_feauters([feature_combinations[c] for c in comb])

        X, Y = features_labels_to_xy(merged_feats, Y_raw)

        # 4. Create pipeline

        pipeline = Pipeline([
            ('scale', MinMaxScaler(feature_range=(-1, 1))),
            ('clf', LinearRegression()),
        ])
        parameters = [
            {
                'clf': (LinearRegression(),),
                'clf__normalize': (True, False),
                # 'clf__random_state': (42,)
            },
             {
                'clf': (BayesianRidge(),),
                'clf__n_iter': (300, 500),
                'clf__lambda_1': (1e-6, 1e-5),
                'clf__lambda_2': (1e-6, 1e-5),
                'clf__alpha_1': (1e-6, 1e-5),
                'clf__alpha_2': (1e-6, 1e-5),
                # 'clf__random_state': (42,)
            },
            {
                'clf': (SVR(),),
                'clf__kernel': ('linear', 'rbf'),
                'clf__C': (0.5, 1.0, 0.0001),
                'clf__gamma': (0.0001, 0.001, 0.01, 0.1),
                # 'clf__random_state': (42,)
            },
            {
                'clf': (RandomForestRegressor(max_features='sqrt'),),
                'clf__n_estimators': (1, 5, 10, 100, 300),
                'clf__random_state': (42,)
            },
            {
                'clf': (KNeighborsRegressor(),),
                'clf__n_neighbors': (3, 5, 7, 9),
            },
        ]
        grid_search = GridSearchCV(pipeline, parameters, cv=5)

        clf = grid_search.fit(X, Y)

        best_estimator = clf.best_estimator_

        # 6. LOOCV
        loo = LeaveOneOut()

        predicted = []
        gold = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            new_fit = best_estimator.fit(X_train, y_train)

            pred = new_fit.predict(X_test)

            predicted.append(pred)
            gold.append(y_test)

        # clas_rep = classification_report(gold, predicted)
        performance = mean_squared_error(gold, predicted)
        print("Pred performance (MSE): ", performance)
        print(clf.best_params_)
        print("Best tuning score(R2): ", clf.best_score_)

        print(X.shape)
        # print("Average: ", np.average(Y))
        average = np.full(Y.shape, np.average(Y))
        baseline = mean_squared_error(gold, average)
        print('Baseline(MSE): {}'.format(baseline))

        print('=====')

        logging('regression_scorring.tsv', comb, clf.best_params_, performance)
