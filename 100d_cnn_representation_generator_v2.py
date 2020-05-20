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

import matplotlib.pyplot as plt

from gensim.scripts.glove2word2vec import glove2word2vec

import pickle
from ast import literal_eval
import glob

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
    print(model.summary())

    return model

X, y = load_data_and_labels()
text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
labels_to_save = y_train

#create class weight dict
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights_d = dict(enumerate(class_weights))
num_labels = len(np.unique(y))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)
# X_train = tokenizer.texts_to_sequences(text_train)
# x_test = tokenizer.texts_to_sequences(text_test)


# saving for later use
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

X_train = tokenizer.texts_to_sequences(text_train)
x_test = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X_train) # longest text in train set
print(maxlen)
# maxlen = 100 # or fixed length for improved efficiency
print('vocabubary size:',vocab_size)
print('max length text:',maxlen)

# pad sequences to uniform length
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
# convert int labels to binary array
y_train = (y_train[:,None] == np.unique(y_train)).astype(int)
y_test = (y_test[:,None] == np.unique(y_test)).astype(int)

embedding_dim = 300
embedding_path = "/Users/asi/connor_asi/embeddings/glove.6B/glove.6B.300d.word2vec.txt"
vocab_size = len(word_index) + 1
max_document_length = max([len(x.split(" ")) for x in text_train])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

print(len(X_train))
def transform(raw_documents):
    """Transform documents to word-id matrix.
    Convert words to ids with vocabulary fitted with fit or the one
    provided in the constructor.
    Args:
        raw_documents: An iterable which yield either str or unicode.
    Yields:
        x: iterable, [n_samples, max_document_length]. Word-id matrix.
    """
    for tokens in _tokenizer(raw_documents):
        word_ids = np.zeros(max_document_length, np.int64)
        for idx, token in enumerate(tokens):
            if idx >= max_document_length:
                break
            word_ids[idx] = vocabulary_.get(token)
        yield word_ids

vocab_processor = transform(max_document_length)

# load the whole embedding into memory
embeddings_index = dict()
print(f'loading vectors from {embedding_path}')
f = open(embedding_path)
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

# pickle.dump(embeddings_index, open( "/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff.pkl", "wb" ) )
# embeddings_index = pickle.load(open("/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff.pkl", "rb"))

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# Check % words with embeddings 
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size) 

pickle.dump(embedding_matrix, open( "/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff_matrix.pkl", "wb" ) )
embedding_matrix = pickle.load(open("/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff_matrix.pkl", "rb"))



model = TextCNN(embedding_matrix, sequence_length=maxlen, num_classes=11, vocab_size=vocab_size, 
        embedding_size=embedding_dim, filter_sizes=[3,4,5], num_filters=50)
extractor = Model(model.input, model.get_layer('extractor').output)
representation = extractor.predict(X_train)

# np.savetxt('100d_utterance_representations.csv', representation, delimiter=',')
# test = np.loadtxt('100d_utterance_representations.csv', delimiter=',')
# print(len(test))
# print(test[0].shape)
# print(test.shape)

# data = np.loadtxt('/Users/asi/connor_asi/cnn_for_100d_utterance_representation/100d_utterance_representations.csv', delimiter=',')      
data = representation
df_path = glob.glob('/Users/asi/connor_asi/spaff_data/utterance_level_transcripts/*')

df_list = []
for path in df_path:
    d = pd.read_csv(path)
    df_list.append(d)

def get_indices(df_list):
    """
    Returns the split indices inside the array.
    """
    indices = [0]
    for df in df_list:
        indices.append(len(df) + indices[-1])
    return indices[1:]

# split the given arr into multiple sections.
sections = np.split(data, get_indices(df_list))
count = 0
for d, s in zip(df_list, sections):
    if len(d) == len(s):
        d['100d'] = s.tolist() # append the section of array to dataframe
        d.to_csv(f'/Users/asi/connor_asi/spaff_data/utterance_with_100d/df_{count}.csv')
        count += 1
    
new_df_path = glob.glob('/Users/asi/connor_asi/spaff_data/utterance_with_100d/*')
new_keys = []
for i in new_df_path:
    new_keys.append(i.split('/')[-1][:-4])

dict_text = {}
dict_labels = {}
dict_speaker = {}
dict_umask = {}
dict_raw_text = {}
for i in new_keys:
    for j in new_df_path:
        if i in j:
            d = pd.read_csv(j)
            d = d[d.speaker_spaff != 7]
            d = d[d.speaker_spaff != 8]
            d = d[d.speaker_spaff != 13]
            d = d[d.speaker_spaff != 14]
            print(d.speaker_spaff.value_counts())

            d['speaker_spaff_'] = d.speaker_spaff.astype('category').cat.codes
            print(d.speaker_spaff_.value_counts())

            def relabel_spaff(d):

                if d.speaker_spaff == 1:
                    x = 0
                elif d.speaker_spaff == 2:
                    x = 1
                elif d.speaker_spaff == 3:
                    x = 2
                elif d.speaker_spaff == 4:
                    x = 3
                elif d.speaker_spaff == 5:
                    x = 4
                elif d.speaker_spaff == 6:
                    x = 5
                elif d.speaker_spaff == 9:
                    x = 6
                elif d.speaker_spaff == 10:
                    x = 7
                elif d.speaker_spaff == 11:
                    x = 8
                elif d.speaker_spaff == 12:
                    x = 9
                elif d.speaker_spaff == 15:
                    x = 10

                return x

            d['relabeled_spaff'] = d.apply(relabel_spaff, axis=1)

            min_speaker = min(d.speaker_label.values)
            def label(df, min_speaker):
                x = 0
                if df.speaker_label == min_speaker:
                    x = 0
                else:
                    x = 1
                return x
            d['speaker_label'] = d.apply(label, axis=1, args=(min_speaker,))
            
            def format(df):
                if df.speaker_label == 0:
                    x = [0,1]
                else:
                    x = [1, 0]
                return x
            d['formated_speaker_label'] = d.apply(format, axis=1)

            d['100d'] = d['100d'].apply(literal_eval)
            d['100d'] = d['100d'].apply(np.array)

            # d['speaker_spaff'] = d.speaker_spaff.apply(literal_eval)
            # d['formated_speaker_label'] = d['formated_speaker_label'].apply(literal_eval)
            dict_text[i] = d['100d'].to_list()
            dict_labels[i] = d.relabeled_spaff.values
            dict_speaker[i] = d.formated_speaker_label.to_list()
            dict_umask[i] = [1] * len(d)
            dict_raw_text[i] = d['word'].to_list()

list_of_dicts = [dict_text, dict_labels, dict_speaker, new_keys, dict_raw_text]

# np.savetxt('100d_X.csv', representation, delimiter=',')
# np.savetxt('y.csv', labels_to_save, delimiter=',')

# pickle.dump(list_of_dicts, open( "spaff_features_300d.pkl", "wb" ) )