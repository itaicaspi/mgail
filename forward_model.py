from collections import OrderedDict
import ops
import tensorflow as tf
import common


class ForwardModel(object):
    def __init__(self, state_size, action_size, encoding_size=50, lr=1e-4):
        self.state_size = state_size
        self.action_size = action_size

        self.arch_params = {
            'input_dim': state_size + action_size,
            'encoding_dim': encoding_size,
            'output_dim': state_size
        }
        self.training_params = {
            'lr': lr
        }

        self.create_variables()

    def forward(self, input):
        state = tf.cast(input[0], tf.float32)
        action = tf.cast(input[1], tf.float32)
        gru_state = tf.cast(input[2], tf.float32)

        # State embedding
        state_embedder1 = tf.nn.relu(tf.matmul(state, self.weights["encoder1_state_weights"]) + self.weights["encoder1_state_biases"])
        gru_state = ops.gru_layer(state_embedder1, gru_state, self.weights, 'gru1')
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

        gru_state = tf.cast(gru_state, tf.float64)
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

    def create_variables(self):
        # set all the necessary weights and biases according to the forward model structure
        self.weights = OrderedDict()
        self.weights.update(ops.gru_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], "gru1"))
        self.weights.update(ops.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'decoder1'))
        self.weights.update(ops.linear_variables(self.arch_params['encoding_dim'], self.arch_params['output_dim'], 'decoder2'))
        self.weights.update(ops.linear_variables(self.state_size, self.arch_params['encoding_dim'], 'encoder1_state'))
        self.weights.update(ops.linear_variables(self.action_size, self.arch_params['encoding_dim'], 'encoder1_action'))
        self.weights.update(ops.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2_state'))
        self.weights.update(ops.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2_action'))
        self.weights.update(ops.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder3'))
        self.weights.update(ops.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder4'))
