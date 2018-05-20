# analysis.oy: Use Trained RNN Model to get sentiment score on new tweets
# Shomik Jain, CAIS++

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import pickle
import numpy as np
import os
import sys
import csv

#Rebuild the Model
MAX_NB_WORDS = 20000
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 109
EMBEDDING_DIM = 100
GLOVE_DIR = "glove.6B"

#tokenizer
tokenizer = pickle.load(open("tokenizer.p","rb"))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#embeddings matrix
print('Preparing embedding matrix.')
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

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


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(64))(embedded_sequences)
x = Dropout(.5)(x)
print(x.shape)
x = Dense(64, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)
model = Model(sequence_input, preds)

#load weights (from training)
filename = "weights-improvement-10-0.4010.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


#Analysis on NEW TWEETS
import re

#iterate over all games/Teams
with open('sentiment.csv', 'w') as csvfile:
    fieldnames = ['game_id', 'team_id', 'avg_neg_conf', 'avg_pos_conf', 'perc_neg_tweets', 'perc_pos_tweets']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for game_id in range(1, 64):
        for team_id in range(1, 3):
            filepath = "data/" + str(game_id) + "-" + str(team_id) + ".txt"

            tweets = []
            with open(filepath, 'r+') as file:
                for tweet in file:
                    #clean up tweet: remove http and .com
                    cleaned_tweet = tweet
                    http_pattern = re.compile('(\S)*http(\S)*')
                    com_pattern = re.compile('(\S)*com(\S)*')
                    http_pattern.sub('', cleaned_tweet)
                    com_pattern.sub('', cleaned_tweet)
                    tweets.append(cleaned_tweet)
            print(len(tweets))
            continue
            sequences = tokenizer.texts_to_sequences(tweets)
            data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            result = model.predict(data)

            #analysis on sentiment RESULTS
            total_neg_conf = 0
            total_pos_conf = 0
            total_pos_tweets = 0
            total_neg_tweets = 0
            total_tweets = len(result)
            print(total_tweets)
            for i in result:
                neg_conf = i[0]
                pos_conf = i[1]

                total_neg_conf = total_neg_conf + neg_conf
                total_pos_conf = total_pos_conf + pos_conf

                threshold = 0.6
                if (neg_conf >= threshold):
                    total_neg_tweets = total_neg_tweets + 1
                if (pos_conf >= threshold):
                    total_pos_tweets = total_pos_tweets + 1

            avg_neg_conf = 0
            avg_pos_conf = 0
            perc_neg_tweets = 0
            perc_pos_tweets = 0
            if (total_tweets != 0):
                avg_neg_conf = total_neg_conf / total_tweets
                avg_pos_conf = total_pos_conf / total_tweets
                perc_neg_tweets = total_neg_tweets / total_tweets
                perc_pos_tweets = total_pos_tweets / total_tweets
            writer.writerow({'game_id': game_id, 'team_id': team_id, 'avg_neg_conf': avg_neg_conf, 'avg_pos_conf': avg_pos_conf, 'perc_neg_tweets': perc_neg_tweets, 'perc_pos_tweets': perc_pos_tweets})
