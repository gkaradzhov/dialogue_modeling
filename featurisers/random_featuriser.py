import csv
import json
from collections import Counter
import ast

import os
import pandas as pd

from read_data import read_solution_annotaions, read_wason_dump

from scipy import stats


if __name__ == '__main__':


    raw_data = read_wason_dump('../data/all_data_20210107/')

    features = []
    for item in raw_data:
        line_feat = [item.identifier, *stats.uniform(0, 2).rvs(128)]
        features.append(line_feat)

    with open('../features/random_vector.tsv', 'w+') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for item in features:
            csv_writer.writerow(item)
