from data_helpers import load_data_and_labels

import pandas as pd
import numpy as np

from keras import layers
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from keras.models import load_model
from keras.utils import to_categorical

# from tensorflow.contrib import learn

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.externals import joblib

# import matplotlib.pyplot as plt

# from gensim.scripts.glove2word2vec import glove2word2vec

import pickle
from ast import literal_eval
import glob
import re



### load trained tokenizer + embedding matrix
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = pickle.load(open("/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff_matrix.pkl", "rb"))


### define model
def TextCNN(embedding_matrix, sequence_length, num_classes, vocab_size, 
            embedding_size, filter_sizes, num_filters):

    sequence_input = Input(shape=(sequence_length,), dtype='int32')

    embedding_layer = Embedding(vocab_size,
                                embedding_size,
                                weights=[embedding_matrix],
                                input_length=sequence_length,
                                trainable=False)

    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    for fsz in filter_sizes:
        x = Conv1D(num_filters, fsz, activation='relu', padding='same')(embedded_sequences)
        x = MaxPooling1D(pool_size=2)(x)
        convs.append(x)

    x = Concatenate(axis=-1)(convs)
    x = Flatten()(x)
    x = Dense(100, activation='relu', name='extractor')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(sequence_input, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


### read in and prepare data
def load_data_and_labels(data_path):

    d = pd.read_csv(data_path)
    d = d[d.speaker_spaff != 7]
    d = d[d.speaker_spaff != 8]
    d = d[d.speaker_spaff != 13]
    d = d[d.speaker_spaff != 14]
    df = d

    # read raw text as data
    data = df.word.values
    # read raw labels and convert to one hot
    # labels = pd.get_dummies(df['speaker_spaff']).values
    labels = df.speaker_spaff.values

    def clean_spaff_str(string):
        """
        SLIGHTLY EDITED: 05-06-2020 for SPAFF
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    prepared_data = []
    for i in data:
        j = clean_spaff_str(i)
        prepared_data.append(j)

    X = prepared_data
    y = labels

    return [X, y, df]


def create_representations(X,y,df):

    maxlen = 77 # maxlen from training
    vocab_size = 8629 # from training
    embedding_dim = 300
    print(X)
    X = tokenizer.texts_to_sequences(X)
    print(X)
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    print(X)


    model = TextCNN(embedding_matrix, sequence_length=maxlen, num_classes=11, vocab_size=vocab_size, 
            embedding_size=embedding_dim, filter_sizes=[3,4,5], num_filters=50)
    extractor = Model(model.input, model.get_layer('extractor').output)
    representation = extractor.predict(X)

    data = list(zip(representation, df.word, y))
   
    pickle.dump(representation, open('test_data.pkl', "wb" ) )

    np.savetxt('sample_rep.csv', representation, delimiter=',')



X, y, df = load_data_and_labels('/Users/asi/connor_asi/spaff_data/utterance_level_transcripts/df_110.csv')
create_representations(X,y,df)