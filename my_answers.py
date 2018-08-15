import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    length = np.size(series) 
    X = [ series[i: i+window_size] for i in range(length-window_size) ]
    y =[ series[i+window_size] for i in range(length-window_size) ]
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']


    text = text.replace('\ufeff', ' ')
    text = text.replace('\u000b', ' ')
    text = text.replace('\r',' ')
    text = text.replace('\t', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\f', ' ')

    text = text.replace('à', 'a')
    text = text.replace('â', 'a')
    text = text.replace('è', 'e')
    text = text.replace('é', 'e')
    text = text.replace('@', ' ')
    text = text.replace('*', ' ')
    text = text.replace('&', ' ')
    text = text.replace('#', ' ')
    text = text.replace('$', ' ')
    text = text.replace('/', ' ')
    text = text.replace('-', ' ')
    text = text.replace('%', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('_', ' ')
    text = text.replace('`', ' ')
    text = text.replace('~', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')
    text = text.replace('|', ' ')
    text = text.replace('^', ' ')
    text = text.replace('=', ' ')
    text = text.replace('\\', ' ')
    text = text.replace('/', ' ')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('+', ' ')
    text = text.replace('\'', ' ')
    text = text.replace('\"', ' ')

    text = text.replace('0', ' ')
    text = text.replace('1', ' ')
    text = text.replace('2', ' ')
    text = text.replace('3', ' ')
    text = text.replace('4', ' ')
    text = text.replace('5', ' ')
    text = text.replace('6', ' ')
    text = text.replace('7', ' ')
    text = text.replace('8', ' ')
    text = text.replace('9', ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    length = len(text)
    inputs = [ text[i: i+window_size] for i in range(0, length-window_size, step_size) ]
    outputs =[ text[i+window_size] for i in range(0, length-window_size, step_size) ]
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model