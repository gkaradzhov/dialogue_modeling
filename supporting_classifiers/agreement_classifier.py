import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle



class Predictor():
    def __init__(self, path='models/agreement.pkl'):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, text):
        dd = self.model.decision_function([text])[0]
        if dd >= -0.001 and len(text.split()) <= 5:
            return 1
        else:
            return 0

if __name__ == '__main__':
    df = pd.read_excel('../data/external/system/affirm.xlsx').to_records(index=False)
    train = [str(d[0]) for d in df]

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1, 2))),
        ('scale', MaxAbsScaler()),
        ('clf', OneClassSVM(kernel='linear'))
    ])

    pipeline.fit(train)

    pkl_filename = "models/agreement.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pipeline, file)