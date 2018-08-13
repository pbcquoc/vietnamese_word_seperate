from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Embedding
import pandas as pd


CHAR_VOCAB = 103
WORD_VOCAB = 5000

MAX_CHARS = 200
MAX_WORDS = 50
EMBED_DIM = 56
LATENT_DIM = 128
OUTPUT_DIM = 56

def tokenize(X, y):
    char_tokenizer = Tokenizer(num_words=CHAR_VOCAB, char_level=True)
    char_tokenizer.fit_on_texts(X)
    X = char_tokenizer.texts_to_sequences(X)
    X1 = pad_sequences(X,  maxlen=MAX_CHARS)

    word_tokenizer = Tokenizer(num_words=WORD_VOCAB)
    word_tokenizer.fit_on_texts(y)
    y = char_tokenizer.texts_to_sequences(y)
    y = pad_sequences(y,  maxlen=MAX_WORDS)
    
    X2 = y.copy()
    X2[:,0] = 0

    return X1, X2, y

def model():
    # embed encoder
    encoder_inputs = Input(shape=(None, MAX_CHARS))
    embed_encoder = Embedding(CHAR_VOCAB, EMBED_DIM, input_length=MAX_CHARS)
    embed_encoder_inputs = embed_encoder(encoder_inputs)

    encoder = LSTM(OUTPUT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(embed_encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, MAX_WORDS))
    embed_decoder = Embedding(WORD_VOCAB, EMBED_DIM)
    embed_decoder_inputs = embed_decoder(decoder_inputs)

    decoder_lstm = LSTM(OUTPUT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(embed_decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(WORD_VOCAB, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define encoder model
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # define inference decoder
    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model

df = pd.read_csv('../data/word_seperate/corpus', header=None)   
X, y = df[0].values, df[1].values

X1, X2, y = tokenize(X, y)
train, infdec, infenc = model()
infdec.summary()
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
train.fit([X1, X2], y, epochs=1)

