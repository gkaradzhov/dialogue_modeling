from scipy import stats
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut

from featurisers.raw_wason_featuriser import get_y
from outcome_prediction.prediction_utils import features_labels_to_xy, logging, \
    read_folder_features
from read_data import read_wason_dump
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # 1. Read Labels
    raw_data = read_wason_dump('../data/all_data_20210107/')
    Y_raw = get_y(raw_data)

    feats = read_folder_features('../features/dialogpt_pretrained/')

    X, Y = features_labels_to_xy(feats, Y_raw)
    # 6. LOOCV
    loo = LeaveOneOut()

    X = np.array(X)
    predicted = []
    gold = []
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, maxlen=400, dtype='float32', padding='pre',
        truncating='pre', value=0.0
    )

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    for train_index, test_index in loo.split(X_padded):

        X_train, X_test = X_padded[train_index], X_padded[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        inputs = tf.keras.Input(shape=(400, 768), dtype="float32")

        x = layers.Bidirectional(layers.LSTM(64, activation="relu"))(inputs)

        x = layers.Dropout(0.5)(x)

        # We add a vanilla hidden layer:
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

        model = tf.keras.Model(inputs, predictions)

        opt = tf.keras.optimizers.Nadam(learning_rate=0.0001)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

        model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=80,
                  verbose=True, batch_size=8, callbacks=my_callbacks)

        pred = model.predict(X_test)
        label = pred[0][0]
        # print("{} ::: {}".format(new_fit.predict_proba(X_test), y_test))
        predicted.append(label)
        gold.append(y_test)
    clas_rep = classification_report(gold, predicted)

    print(clas_rep)
    # print(clf.best_score_)
    performance = accuracy_score(gold, predicted)
    print(performance)

    # print(X.shape)
    mode = stats.mode(Y)
    occs = np.count_nonzero(Y == mode)
    print('Baseline: {}'.format(occs/len(Y)))


    print(model.summary())

    logging('clasification.tsv', 'dialogpt_pretrained', 'PARAMS', performance)


