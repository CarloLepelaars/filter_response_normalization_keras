from keras import backend as K
from keras.engine import Layer
from keras import initializers, regularizers, constraints


class FilterResponseNormalization(Layer):
    """
    Implementation of the Filter Response Normalization (FRN) layer 
    and the Thresholded Linear Unit (TLU).
    
    Source: https://arxiv.org/pdf/1911.09737.pdf
    
    :param eps: An epsilon value to avoid division by zero
    :param weight_initializer: Initializer for the weights.
    :param weight_regularizer: Optional regularizer for the weights.
    :param weight_constraint: Optional constraint for the weights.
    :param bias_initializer: Initializer for the bias.
    :param bias_regularizer: Optional regularizer for the bias.
    :param bias_constraint: Optional constraint for the bias.
    :param threshold_initializer: Initializer for the beta threshold.
    :param threshold_regularizer: Optional regularizer for the threshold.
    :param threshold_constraint: Optional constraint for the threshold.
    """
    def __init__(self, 
                 eps=1e-15, 
                 weight_initializer='ones',
                 weight_regularizer=None,
                 weight_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 threshold_initializer='zeros',
                 threshold_regularizer=None,
                 threshold_constraint=None,
                 **kwargs):
        super(FilterResponseNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.eps = K.variable(eps, dtype=K.floatx())
        self.weight_initializer = initializers.get(weight_initializer)
        self.weight_regularizer = regularizers.get(weight_regularizer)
        self.weight_constraint = constraints.get(weight_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.threshold_constraint = constraints.get(threshold_constraint)
        self.threshold_regularizer = regularizers.get(threshold_regularizer)
        self.threshold_initializer = initializers.get(threshold_initializer)
        
    def build(self, input_shape):
        """
        Intialize weights, bias and threshold variables
        
        :param input_shape: The shape of the input that this layer takes
        """
        shape = self.input_shape[-1:]
        self.weights = self.add_weight(shape=shape,
                                       initializer=self.weight_initializer,
                                       regularizer=self.weight_regularizer,
                                       constraint=self.weight_constraint,
                                       name='weights')
        self.bias = self.add_weight(shape=shape,
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    name='bias')
        self.threshold = self.add_weight(shape=shape,
                                   initializer=self.threshold_initializer,
                                   regularizer=self.threshold_regularizer,
                                   constraint=self.threshold_constraint,
                                   name='threshold')
        super(LayerNormalization, self).build(input_shape)
        
    def call(self, x):
        """
        :param x: Input tensor of shape [NxHxWxC]
        :return: the Filter Response Normalization with Thresholded Linear Unit activation
        """
        # Compute the mean norm of activations per channel.
        nu2 = K.reduce_mean(K.square(x), axis=[1, 2], keepdims=True)
        # Perform FRN
        x = x * K.rsqrt(nu2 + K.abs(eps))
        # Perform TLU activation
        x = self._tlu(x, weights=self.weights, biases=self.bias, threshold=self.threshold)
        return x
    
    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.bias_initializer),
            'gamma_initializer': initializers.serialize(self.weight_initializer),
            'tau_initializer': initializers.serialize(self.threshold_initializer),
            'beta_regularizer': regularizers.serialize(self.bias_regularizer),
            'gamma_regularizer': regularizers.serialize(self.weight_regularizer),
            'tau_regularizer': regularizers.serialize(self.threshold_regularizer),
            'beta_constraint': constraints.serialize(self.bias_constraint),
            'gamma_constraint': constraints.serialize(self.weight_constraint),
            'tau_constraint': constraints.serialize(self.threshold_constraint)
        }
        base_config = super(FilterResponseNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    @staticmethod
    def compute_output_shape(input_shape):
        """
        The output shape will be the same as the input shape.
        """
        return input_shape
    
    @staticmethod
    def _tlu(x, weights, bias, threshold):
        """
        The Thresholded Linear Unit activation (TLU)
        Source: https://arxiv.org/pdf/1911.09737.pdf
        
        :param x: The input transformed by the Filtered Response Normalization
        :param weights: The current weights of the layer
        :param bias: The current biases of the layer
        :param threshold: A learned threshold
        :return: The output of the layer
        """
        return K.maximum(weights * x + bias, threshold)