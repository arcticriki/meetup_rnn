from lasagne.layers import InputLayer, LSTMLayer, DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.init import GlorotNormal

def build(timestep, vocab_size):
    # Input Layer
    l_in = InputLayer(shape=(None, timestep, vocab_size))
    # 2 Hidden LSTM Layers
    l_lstm1 = LSTMLayer(l_in, num_units=10, nonlinearity=rectify)
    l_lstm2 = LSTMLayer(l_lstm1, num_units=10, nonlinearity=rectify, only_return_final=True)
    # Output Layer
    l_out = DenseLayer(l_lstm2, num_units=vocab_size, W=GlorotNormal, nonlinearity=softmax)

    return l_out