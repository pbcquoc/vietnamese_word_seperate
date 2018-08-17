from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Embedding
from keras.layers import TimeDistributed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CHAR_VOCAB = 103
WORD_VOCAB = 5000

MAX_CHARS = 200
MAX_WORDS = 50
EMBED_DIM = 8 
OUTPUT_DIM = 16

def tokenize(X, y):
    char_tokenizer = Tokenizer(num_words=CHAR_VOCAB, char_level=True)
    char_tokenizer.fit_on_texts(X)
    X = char_tokenizer.texts_to_sequences(X)
    X1 = pad_sequences(X,  maxlen=MAX_CHARS, truncating='post')
    
    word_tokenizer = Tokenizer(num_words=WORD_VOCAB)
    word_tokenizer.fit_on_texts(y)
    y = word_tokenizer.texts_to_sequences(y)
    y = pad_sequences(y,  maxlen=MAX_WORDS, padding='post', truncating='post')
    
    X2 = np.pad(y, ((0,0),(1,0)), mode='constant', constant_values=WORD_VOCAB+1)[:,:-1]
    
    char_index = {v:k for k, v in char_tokenizer.word_index.items()}
    word_index = {v:k for k, v in word_tokenizer.word_index.items()}

    return X1, X2, y, char_index, word_index

def model():
    # embed encoder
    encoder_inputs = Input(shape=(MAX_CHARS,))
    embed_encoder = Embedding(CHAR_VOCAB+1, EMBED_DIM, mask_zero=True)
    embed_encoder_inputs = embed_encoder(encoder_inputs)
    
    encoder = LSTM(OUTPUT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(embed_encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(MAX_WORDS, ))
    embed_decoder = Embedding(WORD_VOCAB+2, EMBED_DIM, mask_zero=True)
    embed_decoder_inputs = embed_decoder(decoder_inputs)
    
    decoder_lstm = LSTM(OUTPUT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(embed_decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(WORD_VOCAB, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define encoder model
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # define inference decoder
    decoder_state_input_h = Input(shape=(OUTPUT_DIM,))
    decoder_state_input_c = Input(shape=(OUTPUT_DIM,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(embed_decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model

def index_str(indices, vocab, delimiter=''):
    strchar = delimiter.join([vocab[idx] for idx in indices if idx != 0 and (idx in vocab)])
    return strchar

def predict(infenc, infdec, source):
    #encode
    state = infenc.predict(source)
    current_word = WORD_VOCAB + 1
    output = []
    for t in range(MAX_WORDS):
        target_seq = np.zeros(shape=(len(source), MAX_WORDS))
        target_seq[:, 0] = current_word
        
        yhat, h, c = infdec.predict([target_seq]+state)
        yhat = yhat[:,0, :]
        state = [h, c]
        current_word = np.argmax(yhat, axis=-1)
        output.append(current_word)

    return np.asarray(output).T

df = pd.read_csv('../data/word_seperate/testcorpus', header=None) 
X, y = df[0].values, df[1].values

X1, X2, y, char_index, word_index = tokenize(X, y)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2, y, test_size=0.2, random_state=2018)

print(X1_train.shape, X1_test.shape)

train, infenc, infdec = model()
train.summary()

train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

batch_size = 1 
epoches = 1000
for epoch in range(epoches):
    for batch in range(int(len(y_train)/batch_size)):
        X1_batch = X1_train[batch_size*batch:batch_size*(batch+1), :]
        X2_batch = X2_train[batch_size*batch:batch_size*(batch+1), :]
        y_batch = y_train[batch_size*batch:batch_size*(batch+1), :]
        print(X1_batch, X2_batch, y_batch)
        y_batch = to_categorical(y_batch.flatten(), num_classes=WORD_VOCAB)
        y_batch = y_batch.reshape((-1, MAX_WORDS, WORD_VOCAB))
        
        xent, acc = train.train_on_batch([X1_batch, X2_batch], y_batch)
        if (batch % 20) == 0:
            print(epoch, batch, xent, acc)
            predicts = predict(infenc, infdec, X1_test)
            for i in range(1):
                strpredict = index_str(X1_test[i], char_index,"") +"->"+index_str(predicts[i], word_index, " ")
                print(strpredict)
