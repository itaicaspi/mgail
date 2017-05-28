import tensorflow as tf


def dense(input, input_size, output_size, activation, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse, initializer=tf.random_normal_initializer(stddev=0.15)):
        weights = tf.get_variable('weights', [input_size, output_size])
        biases = tf.get_variable('biases', [output_size])
        output = tf.matmul(input, weights) + biases
        if activation:
            output = activation(output)
        return output


def gru(input, hidden, input_size, hidden_size, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse, initializer=tf.random_normal_initializer(stddev=0.15)):
        Wxr = tf.get_variable('weights_xr', [input_size, hidden_size])
        Wxz = tf.get_variable('weights_xz', [input_size, hidden_size])
        Wxh = tf.get_variable('weights_xh', [input_size, hidden_size])
        Whr = tf.get_variable('weights_hr', [hidden_size, hidden_size])
        Whz = tf.get_variable('weights_hz', [hidden_size, hidden_size])
        Whh = tf.get_variable('weights_hh', [hidden_size, hidden_size])
        br = tf.get_variable('biases_r', [1, hidden_size])
        bz = tf.get_variable('biases_z', [1, hidden_size])
        bh = tf.get_variable('biases_h', [1, hidden_size])

        x, h_ = input, hidden
        r = tf.sigmoid(tf.matmul(x, Wxr) + tf.matmul(h_, Whr) + br)
        z = tf.sigmoid(tf.matmul(x, Wxz) + tf.matmul(h_, Whz) + bz)

        h_hat = tf.tanh(tf.matmul(x, Wxh) + tf.matmul(tf.multiply(r, h_), Whh) + bh)

        output = tf.multiply((1 - z), h_hat) + tf.multiply(z, h_)

        return output
