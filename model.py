import pandas as pd
import numpy as np
import math
import nltk
import re 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pickle
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.python.framework import ops
ops.reset_default_graph()
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 200)
def read_text(filename):
    # open the file
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    file.close()
    return text
def to_pairs(doc):
    lines = doc.strip().split('\n')
    return lines
nepali= read_text("bible.ne")
nepali_words=to_pairs(nepali)
nepali_lines= array(nepali_words)

def max_length(lines):
    list1 = [] 
    for i in range(len(lines)):
        a=(len(lines[i].split(' ')))
        list1.append(a)
    return max(list1)   
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
nep_tokenizer = tokenization(nepali_lines)
nep_vocab_size = len(nep_tokenizer.word_index) + 1

nep_length = 71
print(nep_vocab_size)
english= read_text("english.en")
english_words=to_pairs(english)

nep_length=max_length(nepali_lines)
english_lines= array(english_words)

eng_tokenizer = tokenization(english_lines)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 81
print(eng_length)
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
from sklearn.model_selection import train_test_split
trainx, testx = train_test_split(nepali_lines, test_size=0.5, random_state = 12)
trainy, testy = train_test_split(english_lines,test_size=0.5, random_state = 12)
trainX = encode_sequences(nep_tokenizer, nep_length, trainx)
trainY = encode_sequences(eng_tokenizer, eng_length, trainy)
testX = encode_sequences(nep_tokenizer, nep_length, testx)
testY = encode_sequences(eng_tokenizer, eng_length, testy)

def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model
model = build_model(nep_vocab_size, eng_vocab_size, nep_length, eng_length, 256)
rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(trainX, trainY, epochs=30, batch_size=205,validation_split = 0.5, verbose=1)
model.save('translation_model') 
model = load_model('model')
preds = model.predict(testX[:15])
def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==n:
            return word
    return None
preds_text = []
for i in pred:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
             
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))
print(preds_text)
