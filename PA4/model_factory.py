################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

# Build and return the model here based on the configuration.
from ImageCaptionLSTM import ImageCaptionLSTM
from ImageCaptionLSTM import ImageCaptionVanilla
from ImageCaptionLSTM import ImageCaptionGRU
from vocab import Vocabulary


def get_model(config_data, vocab: Vocabulary):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    model = None
    if model_type == 'LSTM':
        model = ImageCaptionLSTM(hidden_size, embedding_size, vocab)
    elif model_type == 'Vanilla':
        model = ImageCaptionVanilla(hidden_size, embedding_size, vocab)
    elif model_type == 'GRU':
        model = ImageCaptionGRU(hidden_size, embedding_size, vocab)
    else:
        raise Exception('Only support LSTM, Vanilla and GRU, got {}'.format(model_type))
        
    return model
