import csv
from argparse import ArgumentParser

from scipy import stats
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from tensorflow.python.keras.layers import Embedding

from featurisers.raw_wason_featuriser import get_y
from outcome_prediction.prediction_utils import logging, \
    get_raw_features, features_to_arrays
from read_data import read_wason_dump
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Model

def create_tokeniser(raw):
    all_words = []

    for item in raw.values():
        all_words.append(" ".join(item))

    tok = Tokenizer(filters='*')
    tok.fit_on_texts(all_words)

    return tok



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--type", default='lstm')
    parser.add_argument("--dense_1", default='128')
    parser.add_argument("--dense_2", default='64')
    parser.add_argument("--optimiser", default='sgd')
    parser.add_argument("--recurrent_size", default='64')
    parser.add_argument("--cnn_size", default='32')


    args = parser.parse_args()

    # 1. Read Labels
    raw_data = read_wason_dump('../data/all_data_20210107/')
    Y_raw = get_y(raw_data)

    type = get_raw_features('../features/raw_annotations_type.tsv')
    target = get_raw_features('../features/raw_annotations_target.tsv')
    additional = get_raw_features('../features/raw_annotations_additional.tsv')
    sc_turns = get_raw_features('../features/sc_turns.tsv')
    sc_messages = get_raw_features('../features/sc_messages.tsv')
    sol_part = get_raw_features('../features/solution_participation.tsv')

    tokeniser_type = create_tokeniser(type)
    tokeniser_target = create_tokeniser(target)
    tokeniser_additional = create_tokeniser(additional)

    feat_dict_raw = {'type': type, 'target': target,
                     'additional': additional, 'sc_turn': sc_turns,
                     'sc_messages': sc_messages, 'sol_part': sol_part}
    feat_dict_processed, Y, ordering = features_to_arrays(feat_dict_raw, Y_raw)
    # 6. LOOCV

    ordering = np.array(ordering)

    X_type_seqs = tokeniser_type.texts_to_sequences([" ".join(i) for i in feat_dict_processed['type']])
    X_target_seqs = tokeniser_target.texts_to_sequences([" ".join(i) for i in feat_dict_processed['target']])
    X_additional_seqs = tokeniser_additional.texts_to_sequences([" ".join(i) for i in feat_dict_processed['additional']])


    loo = LeaveOneOut()

    predicted = []
    gold = []
    X_type_seqs = np.array(X_type_seqs)
    X_target_seqs = np.array(X_target_seqs)
    X_additional_seqs = np.array(X_additional_seqs)
    X_sc_turns = np.array(feat_dict_processed['sc_turn'], dtype='float32')
    X_sc_messages = np.array(feat_dict_processed['sc_messages'], dtype='float32')
    X_sol_part = np.array(feat_dict_processed['sol_part'], dtype='float32')


    X_type_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        X_type_seqs, padding='pre',
        truncating='pre', value=0, dtype='int32'
    )

    X_target_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        X_target_seqs, padding='pre',
        truncating='pre', value=0, dtype='int32'
    )

    X_additional_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        X_additional_seqs, padding='pre',
        truncating='pre', value=0, dtype='int32'
    )

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
    ]
    counter = 0
    representation = {}
    for train_index, test_index in loo.split(Y):
        print("=================LOOCV number: {}=======".format(counter))
        counter += 1
        X_ind_type, X_test_type = X_type_seqs[train_index], X_type_seqs[test_index]
        X_ind_target, X_test_target = X_target_seqs[train_index], X_target_seqs[test_index]
        X_ind_additional, X_test_additional = X_target_seqs[train_index], X_target_seqs[test_index]
        X_ind_sc_turn, X_test_sc_turn = X_sc_turns[train_index], X_sc_turns[test_index]
        X_ind_sc_messages, X_test_sc_messages = X_sc_messages[train_index], X_sc_messages[test_index]
        X_ind_sol, X_test_sol = X_sol_part[train_index], X_sol_part[test_index]

        y_ind, y_test = Y[train_index], Y[test_index]

        X_train_type, X_val_type, y_train, y_val = train_test_split(X_ind_type, y_ind, test_size=0.2, random_state=42)
        X_train_target, X_val_target = train_test_split(X_ind_type, test_size=0.2, random_state=42)
        X_train_additional, X_val_additional = train_test_split(X_ind_type, test_size=0.2, random_state=42)

        X_train_sc_turn, X_val_sc_turn = train_test_split(X_ind_sc_turn, test_size=0.2, random_state=42)
        X_train_sc_messages, X_val_sc_messages = train_test_split(X_ind_sc_messages, test_size=0.2, random_state=42)
        X_train_sol, X_val_sol = train_test_split(X_ind_sol, test_size=0.2, random_state=42)

        # Type encoder
        input_type = tf.keras.Input(shape=(None,))
        x_type = Embedding(input_dim=len(tokeniser_type.word_index) + 2, output_dim=32)(input_type)

        if args.type == 'cnn':
            x_type = layers.Conv1D(int(args.cnn_size), 2, activation='relu')(x_type)
            x_type = layers.GlobalMaxPool1D()(x_type)
            x_type = layers.Dropout(0.5)(x_type)
        elif args.type == 'gru':
            x_type = layers.Bidirectional(layers.GRU(int(args.recurrent_size), activation="relu"))(x_type)
            x_type = layers.Dropout(0.5)(x_type)
        else:
            x_type = layers.Bidirectional(layers.LSTM(int(args.recurrent_size), activation="relu"))(x_type)
            x_type = layers.Dropout(0.5)(x_type)


        # Target encoder
        input_target = tf.keras.Input(shape=(None,))
        x_target = Embedding(input_dim=len(tokeniser_target.word_index) + 2, output_dim=32)(input_target)
        if args.type == 'cnn':
            x_target = layers.Conv1D(int(args.cnn_size), 3, activation='relu')(x_target)
            x_target = layers.GlobalMaxPool1D()(x_target)
            x_target = layers.Dropout(0.5)(x_target)

        elif args.type == 'gru':
            x_target = layers.Bidirectional(layers.GRU(int(args.recurrent_size), activation="relu"))(x_target)
            x_target = layers.Dropout(0.5)(x_target)
        else:
            x_target = layers.Bidirectional(layers.LSTM(int(args.recurrent_size), activation="relu"))(x_target)
            x_target = layers.Dropout(0.5)(x_target)


        # Additional encoder
        input_additional = tf.keras.Input(shape=(None,))
        x_additional = Embedding(input_dim=len(tokeniser_additional.word_index) + 2, output_dim=32)(input_additional)

        if args.type == 'cnn':
            x_additional = layers.Conv1D(int(args.cnn_size), 3, activation='relu')(x_additional)
            x_additional = layers.GlobalMaxPool1D()(x_additional)
            x_additional = layers.Dropout(0.5)(x_additional)

        elif args.type == 'gru':
            x_additional = layers.Bidirectional(layers.GRU(int(args.recurrent_size), activation="relu"))(x_additional)
            x_additional = layers.Dropout(0.5)(x_additional)
        else:
            x_additional = layers.Bidirectional(layers.LSTM(int(args.recurrent_size), activation="relu"))(x_additional)
            x_additional = layers.Dropout(0.5)(x_additional)


        # Features

        input_sc_turn = tf.keras.Input(shape=(6,))
        dense_sc_turn = layers.Dense(64, activation="elu")(input_sc_turn)

        input_sc_messages = tf.keras.Input(shape=(15,))
        dense_sc_messages = layers.Dense(64, activation="elu")(input_sc_messages)

        input_sol = tf.keras.Input(shape=(13,))
        dense_sol = layers.Dense(64, activation="elu")(input_sol)


        attention_1 = layers.Attention()([x_type, x_target])
        attention_2 = layers.Attention()([x_target, x_additional])

        concat = layers.Concatenate()([x_type, x_target, x_additional, attention_1, attention_2,
                                       dense_sc_turn, dense_sc_messages, dense_sol])

        concat = layers.Dense(int(args.dense_1), activation="elu")(concat)

        concat = layers.Dropout(0.5)(concat)
        concat = layers.Dense(int(args.dense_2), activation="sigmoid", name='representation')(concat)

        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(concat)

        model = tf.keras.Model([input_type, input_target, input_additional, input_sc_turn, input_sc_messages, input_sol], predictions)

        model.compile(loss="binary_crossentropy", optimizer=args.optimiser, metrics=["accuracy"])

        model.fit(x=[X_train_type, X_train_target, X_train_additional, X_train_sc_turn, X_train_sc_messages, X_train_sol],
                  y=y_train,
                  validation_data=([X_val_type, X_val_target, X_val_additional, X_val_sc_turn, X_val_sc_messages, X_val_sol], y_val),
                  epochs=100, validation_batch_size=2,
                  verbose=True, batch_size=8, callbacks=my_callbacks, use_multiprocessing=True, workers=8)

        pred = model.predict([X_test_type, X_test_target, X_test_additional, X_test_sc_turn, X_test_sc_messages, X_test_sol])
        label = pred[0][0]
        # print("{} ::: {}".format(new_fit.predict_proba(X_test), y_test))
        predicted.append(round(label))
        gold.append(y_test)

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)

        intermediate_output = intermediate_layer_model.predict([X_test_type, X_test_target, X_test_additional, X_test_sc_turn, X_test_sc_messages, X_test_sol])

        representation[ordering[test_index][0]] = intermediate_output[0]

    clas_rep = classification_report(gold, predicted)
    print(str(args))
    print(clas_rep)
    # print(clf.best_score_)
    performance = accuracy_score(gold, predicted)
    print(performance)

    # print(X.shape)
    mode = stats.mode(Y)
    occs = np.count_nonzero(Y == mode)
    print('Baseline: {}'.format(occs/len(Y)))


    print(model.summary())

    logging('clasification_nn_annotations.tsv', ['dialogue-anns'], str(args), performance)


    with open('representation.tsv', 'w') as wf:
        csv_writer = csv.writer(wf, delimiter='\t')
        for key, item in representation.items():
            csv_writer.writerow([key, *item])



