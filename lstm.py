from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

NUM_WORDS = 103
MAX_SEQUENCE_LENGTH = 200
MAX_WORDS = 50

def tokenize(X, y):
    char_tokenizer = Tokenizer(num_words=103, char_level=True)
    char_tokenizer.fit_on_texts(X)
    X = char_tokenizer.texts_to_sequences(X)
    X = pad_sequences(X,  maxlen=MAX_SEQUENCE_LENGTH)

    word_tokenizer = Tokenizer(num_words=5000)
    word_tokenizer.fit_on_texts(y)
    y = char_tokenizer.texts_to_sequences(y)
    y = pad_sequences(y,  maxlen=MAX_WORDS)

df = pd.read_csv('../data/word_seperate/corpus', header=None)   
X, y = df[0].values, df[1].values

X, y = tokenize(X, y)
