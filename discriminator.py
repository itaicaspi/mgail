from collections import OrderedDict
import tensorflow as tf
import ops

class Discriminator(object):
    def __init__(self, in_dim, out_dim, size, lr, do_keep_prob, weight_decay):
        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'do_keep_prob': do_keep_prob
        }

        self.solver_params = {
            'lr': lr,
            'weight_decay': weight_decay
        }

        self.create_variables()

    def forward(self, state, action):
        concat = tf.concat(axis=1, values=[state, action])
        h0 = tf.nn.relu(tf.matmul(concat, self.weights["dense0_weights"]) + self.weights["dense0_biases"])
        h1 = tf.nn.relu(tf.matmul(h0, self.weights["dense1_weights"]) + self.weights["dense1_biases"])
        relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
        d = tf.matmul(relu1_do, self.weights["dense2_weights"]) + self.weights["dense2_biases"]

        return d

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for k, v in self.weights.items() if 'weights' in k])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values())

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads

    def train(self, objective):
        self.loss = objective
        self.minimize = self.backward(self.loss)

    def create_variables(self):
        # we create all the weights and biases once and reuse them between graph runs
        self.weights = OrderedDict()
        self.weights.update(ops.linear_variables(self.arch_params['in_dim'], self.arch_params['n_hidden_0'], 'dense0'))
        self.weights.update(ops.linear_variables(self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], 'dense1'))
        self.weights.update(ops.linear_variables(self.arch_params['n_hidden_1'], self.arch_params['out_dim'], 'dense2'))
