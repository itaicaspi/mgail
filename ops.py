import tensorflow as tf


def dense(input, output_size, activation, weights_stddev):
    assert len(input.shape) == 2
    input_size = input.shape[-1].value
    weights = tf.Variable(tf.random_normal([input_size, output_size], stddev=weights_stddev))
    biases = tf.Variable(tf.random_normal([output_size], stddev=weights_stddev))
    output = tf.matmul(input, weights) + biases
    if activation:
        output = activation(output)
    return output, weights, biases