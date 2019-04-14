from src import normilized_dense
from keras import models, layers


def get_cnn(outputs, is_norm, activation):
    model = models.Sequential()

    model.add(layers.Convolution2D(32, 3, padding='same',
                                   input_shape=(32, 32, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Convolution2D(32, 3))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Convolution2D(64, 3, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Convolution2D(64, 3))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.5))
    model.add(normilized_dense.NormilizedDense(outputs, is_norm=is_norm))
    if activation is not None:
        model.add(layers.Activation(activation))
    return model
