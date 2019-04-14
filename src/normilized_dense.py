import tensorflow as tf

from keras.engine import InputSpec
from keras.layers import Layer
from keras import backend as K
from keras import initializers, regularizers, optimizers, constraints


class NormilizedDense(Layer):
    def __init__(self, units, is_norm=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NormilizedDense, self).__init__(**kwargs)
        self.units = units
        self.is_norm = is_norm
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs, **kwargs):
        if self.is_norm:
            inputs = tf.nn.l2_normalize(inputs, axis=1)
            self.kernel = tf.nn.l2_normalize(self.kernel, axis=0)
        dis_cosin = tf.matmul(inputs, self.kernel)
        return dis_cosin
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
