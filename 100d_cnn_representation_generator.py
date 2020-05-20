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
print(X)# Split train & test
text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
# raw_text_train, raw_text_test, raw_y_train, raw_y_test = train_test_split(raw_text, y, test_size=0.05, random_state=42)


#create class weight dict
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights_d = dict(enumerate(class_weights))
num_labels = len(np.unique(y))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_train = tokenizer.texts_to_sequences(text_train)
print(X_train)
x_test = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X) # longest text in train set
# maxlen = 100 # or fixed length for improved efficiency
print('vocabubary size:',vocab_size)
print('max length text:',maxlen)

# pad sequences to uniform length
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
# X = pad_sequences(X, padding='post', maxlen=maxlen)

# convert int labels to binary array
y_train = (y_train[:,None] == np.unique(y_train)).astype(int)
y_test = (y_test[:,None] == np.unique(y_test)).astype(int)
# y = (y[:,None] == np.unique(y)).astype(int)

embedding_dim = 300
embedding_path = "/Users/asi/connor_asi/embeddings/glove.6B/glove.6B.300d.word2vec.txt"
vocab_size = len(word_index) + 1
max_document_length = max([len(x.split(" ")) for x in text_train])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

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

vocab_processor = transform(maxlen)

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

pickle.dump(embeddings_index, open( "/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff.pkl", "wb" ) )
embeddings_index = pickle.load(open("/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff.pkl", "rb"))

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


model = TextCNN(embedding_matrix, sequence_length=maxlen, num_classes=11, vocab_size=vocab_size, 
        embedding_size=embedding_dim, filter_sizes=[3,4,5], num_filters=50)
model.fit(X_train, y_train)
extractor = Model(model.input, model.get_layer('extractor').output)

representation = extractor.predict(X_train)
print(len(representation))
print(representation[0].shape)
print(representation.shape)

np.savetxt('100d_utterance_representations.csv', representation, delimiter=',')

test = np.loadtxt('100d_utterance_representations.csv', delimiter=',')
print(len(test))
print(test[0].shape)
print(test.shape)

# zipped_data = list(zip(test, raw_text_train, y_train))
# df = pd.DataFrame(zipped_data)
# print(df)
# df.to_csv('utterances_with_100d.csv')