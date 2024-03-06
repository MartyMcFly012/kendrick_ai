import random
import pickle

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Activation
from tensorflow.keras.optimizers import RMSprop

text=''
with open('kendrick.txt') as file:
    text = file.readlines()
text = " ".join(text)

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text.lower())

unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

n_words = 10
input_words = []
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i+n_words])
    next_words.append(tokens[i+n_words])

X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=np.int8)  # Change dtype to np.int8
y = np.zeros((len(next_words), len(unique_tokens)), dtype=np.int8)  # Change dtype to np.int8

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.01),metrics='accuracy')

model.fit(X,y,batch_size=128,epochs=10,shuffle=True)

def pred_next_word(input, n_best):
    input = input.lower()
    for i, word in enumerate(input.split()):
        X[0,i,unique_token_index[word]]
    predictions=model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]