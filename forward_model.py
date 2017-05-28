from collections import OrderedDict
import ops
import tensorflow as tf
import common


class ForwardModel(object):
    def __init__(self, state_size, action_size, encoding_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.encoding_size = encoding_size

        self.lr = lr

    def forward(self, input, reuse=False):
        with tf.variable_scope('forward_model'):
            state = tf.cast(input[0], tf.float32)
            action = tf.cast(input[1], tf.float32)
            gru_state = tf.cast(input[2], tf.float32)

            # State embedding
            state_embedder1 = ops.dense(state, self.state_size, self.encoding_size, tf.nn.relu, "encoder1_state", reuse)
            gru_state = ops.gru(state_embedder1, gru_state, self.encoding_size, self.encoding_size, 'gru1', reuse)
            state_embedder2 = ops.dense(gru_state, self.encoding_size, self.encoding_size, tf.sigmoid, "encoder2_state", reuse)

            # Action embedding
            action_embedder1 = ops.dense(action, self.action_size, self.encoding_size, tf.nn.relu, "encoder1_action", reuse)
            action_embedder2 = ops.dense(action_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid, "encoder2_action", reuse)

            # Joint embedding
            joint_embedding = tf.multiply(state_embedder2, action_embedder2)

            # Next state prediction
            hidden1 = ops.dense(joint_embedding, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder3", reuse)
            hidden2 = ops.dense(hidden1, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder4", reuse)
            hidden3 = ops.dense(hidden2, self.encoding_size, self.encoding_size, tf.nn.relu, "decoder1", reuse)
            next_state = ops.dense(hidden3, self.encoding_size, self.state_size, None, "decoder2", reuse)

            gru_state = tf.cast(gru_state, tf.float64)

            return next_state, gru_state

    def backward(self, loss):
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward_model')

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads

    def train(self, objective):
        self.loss = objective
        self.minimize = self.backward(self.loss)
