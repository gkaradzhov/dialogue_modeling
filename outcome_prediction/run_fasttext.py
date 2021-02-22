import csv
from sklearn.metrics import classification_report, roc_auc_score
from outcome_prediction.prediction_utils import logging
import fasttext

if __name__ == '__main__':

    predicted = []
    gold = []
    representations = []
    for i in range(0, 102):
        X_train = []
        X_test = []
        y_test = []

        fname_train = '../features/fast_text/sc_train_{}.txt'.format(i)
        fname_test = '../features/fast_text/sc_test_{}.tsv'.format(i)

        with open(fname_test, 'r') as rf:
            tsv_reader = csv.reader(rf, delimiter='\t')
            for item in tsv_reader:
                identifier = item[0]
                y_test = item[1]
                X_test = item[2]

        model = fasttext.train_supervised(fname_train, epoch=500, ws=10, dim=20, wordNgrams=1)

        pred = model.predict(X_test)
        predicted.append(round(pred[1][0]))
        gold.append(int(y_test))

        repres = model.get_sentence_vector(X_test)
        representations.append([identifier, *repres])
        print()

    clas_rep = classification_report(gold, predicted)

    print(clas_rep)
    # print(clf.best_score_)
    performance = roc_auc_score(gold, predicted)
    print(performance)

    # print(X.shape)
    # mode = stats.mode(Y)
    # occs = np.count_nonzero(Y == mode)
    # baseline = roc_auc_score(gold, [mode[0]]*len(Y))
    # print('Baseline: {}'.format(baseline))
    logging('clasification_roc_AUC_fast_text_class2.tsv', ['fasttext'], '', performance)

    with open('../features/fast_text_representation_sc.tsv', 'w') as wf:
        tsv_writer = csv.writer(wf, delimiter='\t')
        for item in representations:
            tsv_writer.writerow([str(s) for s in item])
