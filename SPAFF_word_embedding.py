from data_helpers import load_data_and_labels

# import pandas as pd
import numpy as np

# from keras import layers
# from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
# from keras.models import load_model
from keras.utils import to_categorical

# # from tensorflow.contrib import learn

# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.utils import class_weight
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.externals import joblib

# import matplotlib.pyplot as plt

# from gensim.scripts.glove2word2vec import glove2word2vec

import pickle
# from ast import literal_eval
# import glob


X, y = load_data_and_labels()
labels_to_save = y
# convert int labels to binary array
y = (y[:,None] == np.unique(y)).astype(int)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)

# # saving for later use
# with open('tokenizer.pkl', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # loading
# with open('tokenizer.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X) # longest text in train set
# maxlen = 100 # or fixed length for improved efficiency
print('vocabubary size:',vocab_size)
print('max length text:',maxlen)

# pad sequences to uniform length
X_tokenized_pad = pad_sequences(X_tokenized, padding='post', maxlen=maxlen)

embedding_dim = 300
embedding_path = "/Users/asi/connor_asi/embeddings/glove.6B/glove.6B.300d.word2vec.txt"
max_document_length = max([len(x.split(" ")) for x in X])

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
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

assert len(embedding_matrix) == vocab_size

# Check % words with embeddings 
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print('Percent of words with embedding', nonzero_elements / vocab_size) 

### save and load embedding matrix
pickle.dump(embedding_matrix, open( "/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff_matrix.pkl", "wb" ) )
embedding_matrix = pickle.load(open("/Users/asi/connor_asi/embeddings/300d_glove_trained_on_spaff_matrix.pkl", "rb"))

