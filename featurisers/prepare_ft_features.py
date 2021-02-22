import csv

from sklearn.model_selection import LeaveOneOut

from featurisers.raw_wason_featuriser import get_y
from read_data import read_wason_dump
import numpy as np


if __name__ == '__main__':

    annotations = []
    with open('../features/raw_annotations_additional.tsv', 'r') as rf:
        csv_reader = csv.reader(rf, delimiter='\t')
        for item in csv_reader:
            annotations.append(item)
    annotations = np.array(annotations)
    raw_data = read_wason_dump('../data/all/')

    Y = get_y(raw_data)

    loo = LeaveOneOut()

    id = 0
    for train_index, test_index in loo.split(annotations):
        fname_train = '../features/fast_text/additional_train_{}.txt'.format(id)
        fname_test = '../features/fast_text/additional_test_{}.tsv'.format(id)
        id += 1
        X_train, X_test = annotations[train_index], annotations[test_index]

        with open(fname_train, 'w') as wf:
            for item in X_train:
                y = Y[item[0]]
                wf.write('__label__' + str(y) + ' ' + " ".join(item[1:]) + '\n')

        with open(fname_test, 'w') as wf:
            tsv_writer = csv.writer(wf, delimiter='\t')
            for item in X_test:
                y = Y[item[0]]
                tsv_writer.writerow([
                    str(item[0]),
                    str(y),
                    " ".join(item[1:])])