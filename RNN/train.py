# train.py: RNN Model trained on Sentiment140 dataset
# Shomik Jain, USC CAIS++
# References: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import numpy as np
import os
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

texts = pickle.load(open("/hdd/datasets/sentiment140/train_update.p","rb"))
labels = []
samples = []

# max_length = 0
# gather samples
for text in texts:
    labels.append(text[0])
    samples.append(text[1])
texts = samples #renaming

#iterate through labels to change them to traditional 0 and 1
for i in range(len(labels)):
    if(labels[i] == '4'):
        labels[i] = 1
    else:
        labels[i] = 0

#initialize parameters
MAX_NB_WORDS = 20000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = .1
MAX_SEQUENCE_LENGTH = 109
EMBEDDING_DIM = 100
GLOVE_DIR = "/hdd/datasets/glove.6B"

tokenizer = pickle.load(open("tokenizer.p","rb"))
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

#this part can change, probably best to use rnn when dealing with variable length data
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(64))(embedded_sequences)
x = Dropout(.5)(x)
print(x.shape)
x = Dense(64, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

filepath="./weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min')
earlystop = EarlyStopping(monitor='val_loss', patience =3)
tensorboard = TensorBoard()
callbacks_list = [checkpoint, earlystop, tensorboard]

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)
