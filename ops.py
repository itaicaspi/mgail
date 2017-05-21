import tensorflow as tf

def linear_variables(input_size, output_size, name):
    weights = {}
    weights[name+'_weights'] = weight_variable([input_size, output_size])
    weights[name+'_biases'] = bias_variable([1, output_size])
    return weights

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.15, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape, stddev=0.15, dtype=tf.float32)
    return tf.Variable(initial)

def gru_layer(input, hidden, weights, name):
    x, h_ = input, hidden
    r = tf.sigmoid(tf.matmul(x, weights[name+'_Wxr']) + tf.matmul(h_, weights[name+'_Whr']) + weights[name+'_br'])
    z = tf.sigmoid(tf.matmul(x, weights[name+'_Wxz']) + tf.matmul(h_, weights[name+'_Whz']) + weights[name+'_bz'])

    h_hat = tf.tanh(
        tf.matmul(x, weights[name+'_Wxh']) + tf.matmul(tf.multiply(r, h_), weights[name+'_Whh']) + weights[name+'_bh'])

    output = tf.multiply((1 - z), h_hat) + tf.multiply(z, h_)

    return output

def gru_variables(hidden_size, input_size, name):
    weights = {}
    weights[name+'_Wxr'] = weight_variable([input_size, hidden_size])
    weights[name+'_Wxz'] = weight_variable([input_size, hidden_size])
    weights[name+'_Wxh'] = weight_variable([input_size, hidden_size])
    weights[name+'_Whr'] = weight_variable([hidden_size, hidden_size])
    weights[name+'_Whz'] = weight_variable([hidden_size, hidden_size])
    weights[name+'_Whh'] = weight_variable([hidden_size, hidden_size])
    weights[name+'_br'] = bias_variable([1, hidden_size])
    weights[name+'_bz'] = bias_variable([1, hidden_size])
    weights[name+'_bh'] = bias_variable([1, hidden_size])
    return weights