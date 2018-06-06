from keras.models import Sequential
from keras.layers import Reshape, Dense, LSTM


def build(timestep, vocab_size):
    model = Sequential()
    # Input Layer
    model.add(Reshape(None, timestep, vocab_size))
    # 2 Hidden LSTM Layers
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(LSTM(units=10, activation='relu', return_sequences=False))
    # Output Layer
    model.add(Dense(units=vocab_size, activation='softmax'))


def compile(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model