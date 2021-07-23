import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec


class Rescaling(Layer):
    """
    Layer to apply rescaling during training and inference on every input element.
    """

    def __init__(self, scale, name=None, **kwargs):
        super(Rescaling, self).__init__(name=name, **kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        return tf.cast(inputs, dtype=self.dtype) * tf.cast(self.scale, dtype=self.dtype)

    def get_config(self):
        # create new config data
        conf = {'scale': self.scale}
        # get the base config from the Layer super class
        base_conf = super(Rescaling, self).get_config()
        # merge the base config and the new data into a new configuration
        return dict(**conf, **base_conf)


class L2Normalization(Layer):
    """
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.
    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    """

    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.gamma = None

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.gamma = tf.keras.backend.variable(20 * tf.ones((input_shape[3],)),
                                               name='{}_gamma'.format(self.name))
        self._trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = tf.keras.backend.l2_normalize(x, 3)
        return output * self.gamma
