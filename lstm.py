from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


CHAR_VOCAB = 103
WORD_VOCAB = 5000

MAX_CHARS = 200
MAX_WORDS = 50
EMBED_DIM = 56
LATENT_DIM = 128

def tokenize(X, y):
    char_tokenizer = Tokenizer(num_words=CHAR_VOCAB, char_level=True)
    char_tokenizer.fit_on_texts(X)
    X = char_tokenizer.texts_to_sequences(X)
    X = pad_sequences(X,  maxlen=MAX_CHARS)

    word_tokenizer = Tokenizer(num_words=WORD_VOCAB)
    word_tokenizer.fit_on_texts(y)
    y = char_tokenizer.texts_to_sequences(y)
    y = pad_sequences(y,  maxlen=MAX_WORDS)
    
    return X, y

def model():
    encoder_inputs = Input(shape=(None, MAX_CHARS))
    embed = Embedding(CHAR_VOCAB, EMBED_DIM)

    encoder = LSTM(MAX_CHARS, return_state=True)

df = pd.read_csv('../data/word_seperate/corpus', header=None)   
X, y = df[0].values, df[1].values

X, y = tokenize(X, y)
