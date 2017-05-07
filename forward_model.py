from collections import OrderedDict

import tensorflow as tf

import common


class ForwardModel(object):
    def __init__(self, state_size, action_size, encoding_size=50):
        self.state_size = state_size
        self.action_size = action_size
        self.multi_layered_encoder = True
        self.separate_encoders = True

        self.arch_params = {
            'input_dim': state_size + action_size,
            'encoding_dim': encoding_size,
            'small_encoding_dim': 5,
            'output_dim': state_size
        }
        self.training_params = {
            'lr': 1e-4
        }

        # set all the necessary weights and biases according to the forward model structure
        self.weights = OrderedDict()

        self.weights.update(self.gru_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], "gru1"))

        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'decoder1'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['output_dim'], 'decoder2'))

        self.weights.update(self.linear_variables(state_size, self.arch_params['encoding_dim'], 'encoder1_state'))
        self.weights.update(self.linear_variables(action_size, self.arch_params['encoding_dim'], 'encoder1_action'))

        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2_state'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2_action'))

        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder3'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder4'))


    def gru_variables(self, hidden_size, input_size, name):
        weights = OrderedDict()
        weights[name+'_Wxr'] = self.weight_variable([input_size, hidden_size])
        weights[name+'_Wxz'] = self.weight_variable([input_size, hidden_size])
        weights[name+'_Wxh'] = self.weight_variable([input_size, hidden_size])
        weights[name+'_Whr'] = self.weight_variable([hidden_size, hidden_size])
        weights[name+'_Whz'] = self.weight_variable([hidden_size, hidden_size])
        weights[name+'_Whh'] = self.weight_variable([hidden_size, hidden_size])
        weights[name+'_br'] = self.bias_variable([1, hidden_size])
        weights[name+'_bz'] = self.bias_variable([1, hidden_size])
        weights[name+'_bh'] = self.bias_variable([1, hidden_size])
        return weights

    def bn_variables(self, size, name):
        weights = OrderedDict()
        weights[name+'_mean'] = tf.Variable(tf.constant(0.0, shape=size))
        weights[name +'_variance'] = tf.Variable(tf.constant(1.0, shape=size))
        weights[name + '_offset'] = tf.Variable(tf.constant(0.0, shape=size))
        weights[name + '_scale'] = tf.Variable(tf.constant(1.0, shape=size))
        return weights

    def tensor_linear_variables(self, input_width, input_depth, output_width, name):
        weights = OrderedDict()
        self.weights[name+'_weights'] = self.weight_variable([input_depth, input_width, output_width])
        self.weights[name+'_biases'] = self.bias_variable([input_depth, 1, output_width])
        return weights

    def linear_variables(self, input_size, output_size, name):
        weights = OrderedDict()
        self.weights[name+'_weights'] = self.weight_variable([input_size, output_size])
        self.weights[name+'_biases'] = self.bias_variable([1, output_size])
        return weights

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def gru_layer(self, input, hidden, weights, name):
        x, h_ = input, hidden
        r = tf.sigmoid(tf.matmul(x, weights[name+'_Wxr']) + tf.matmul(h_, weights[name+'_Whr']) + weights[name+'_br'])
        z = tf.sigmoid(tf.matmul(x, weights[name+'_Wxz']) + tf.matmul(h_, weights[name+'_Whz']) + weights[name+'_bz'])

        h_hat = tf.tanh(
            tf.matmul(x, weights[name+'_Wxh']) + tf.matmul(tf.multiply(r, h_), weights[name+'_Whh']) + weights[name+'_bh'])

        output = tf.multiply((1 - z), h_hat) + tf.multiply(z, h_)

        return output

    def forward(self, input):
        state = tf.cast(input[0], tf.float32)
        action = tf.cast(input[1], tf.float32)
        gru_state = tf.cast(input[2], tf.float32)

        # State embedding
        # state_embedder1 = tf.layers.dense(state, self.arch_params['encoding_dim'], tf.nn.relu,
        #                                   kernel_initializer=tf.truncated_normal_initializer(0, 0.1),
        #                                   bias_initializer=tf.constant_initializer(0.1))
        # self.weights['a'] = state_embedder1.
        state_embedder1 = tf.nn.relu(tf.matmul(state, self.weights["encoder1_state_weights"]) + self.weights["encoder1_state_biases"])
        gru_state = self.gru_layer(state_embedder1, gru_state, self.weights, 'gru1')
        state_embedder2 = tf.sigmoid(tf.matmul(gru_state, self.weights["encoder2_state_weights"]) + self.weights["encoder2_state_biases"])

        # Action embedding
        action_embedder1 = tf.nn.relu(tf.matmul(action, self.weights["encoder1_action_weights"]) + self.weights["encoder1_action_biases"])
        action_embedder2 = tf.sigmoid(tf.matmul(action_embedder1, self.weights["encoder2_action_weights"]) + self.weights["encoder2_action_biases"])

        # Joint embedding
        joint_embedding = tf.multiply(state_embedder2, action_embedder2)

        # Next state prediction
        hidden1 = tf.nn.relu(tf.matmul(joint_embedding, self.weights["encoder3_weights"]) + self.weights["encoder3_biases"])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.weights["encoder4_weights"]) + self.weights["encoder4_biases"])
        hidden3 = tf.nn.relu(tf.matmul(hidden2, self.weights["decoder1_weights"]) + self.weights["decoder1_biases"])
        next_state = tf.matmul(hidden3, self.weights["decoder2_weights"]) + self.weights["decoder2_biases"]

        gru_state = tf.cast(gru_state, tf.float32)
        return next_state, gru_state

    def backward(self, loss):

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.training_params['lr'])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values())
        mean_abs_grad, mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, mean_abs_grad, mean_abs_w

    def train(self, objective):
        self.loss = objective
        self.minimize, self.mean_abs_grad, self.mean_abs_w = self.backward(self.loss)
        self.loss_summary = tf.summary.scalar('loss_t', objective)

