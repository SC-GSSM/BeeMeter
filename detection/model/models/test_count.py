import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, DepthwiseConv2D

input_layer = Input(shape=(300, 300, 3))
x = Conv2D(filters=10, kernel_size=3, padding='same', use_bias=True)(input_layer)
x = DepthwiseConv2D(kernel_size=3, padding='same', use_bias=True)(x)
model = tf.keras.Model(inputs=input_layer, outputs=x)
model.summary()
total_cost = 0
total_param = 0
for l in model.layers:
    if type(l) == Conv2D or type(l) == DepthwiseConv2D:
        _, input_h, input_w, input_c = l.input_shape
        _, output_h, output_w, output_c = l.output_shape
        kernel_h, kernel_w = l.kernel_size
        output_c = l.filters if type(l) == Conv2D else 1
        if l.bias is not None:
            bias_shape = l.bias.shape[0]
        else:
            bias_shape = 0

        total_cost += output_h * output_w * kernel_h * kernel_w * input_c * output_c  # add bias?
        total_param += kernel_w * kernel_h * input_c * output_c + bias_shape

print("Total cost: {0:,}".format(total_cost))
print(total_param)
