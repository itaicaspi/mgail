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

    def forward(self, state, action, reuse=False):

        with tf.variable_scope('discriminator'):
            concat = tf.concat(axis=1, values=[state, action])
            h0 = ops.dense(concat, self.arch_params['in_dim'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0', reuse)
            h1 = ops.dense(h0, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1', reuse)
            relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
            d = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)

        return d

    def backward(self, loss):
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(w) for w in self.weights if 'weights' in w.name])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads

    def train(self, objective):
        self.loss = objective
        self.minimize = self.backward(self.loss)
