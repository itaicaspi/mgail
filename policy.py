from collections import OrderedDict

import tensorflow as tf


class Policy(object):
    def __init__(self, in_dim, out_dim, size, lr, w_std, do_keep_prob, n_accum_steps, weight_decay):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'do_keep_prob': do_keep_prob
        }

        self.solver_params = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weights_stddev': w_std,
            'n_accum_steps': n_accum_steps,
        }

        self._init_layers()

    def forward(self, state):
        '''
        state: vector
        '''

        h0 = tf.nn.xw_plus_b(state, self.weights['w0'], self.biases['b0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['w1'], self.biases['b1'], name='h1')
        relu1 = tf.nn.relu(h1)

        relu1_do = tf.nn.dropout(relu1, self.arch_params['do_keep_prob'])

        a = tf.nn.xw_plus_b(relu1_do, self.weights['wc'], self.biases['bc'], name='a')

        return a

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.weights.values()])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values() + self.biases.values())

        grads = [g for g, v in grads_and_vars]

        variables = [v for g, v in grads_and_vars]

        # gradient clipping
        grads = [tf.clip_by_value(g, -2, 2) for g in grads]

        # accumulate the grads
        accum_grads_op = []
        for i, accum_grad in enumerate(self.accum_grads.values()):
            accum_grads_op.append(accum_grad.assign_add(grads[i]))

        # pack accumulated gradient and vars back in grads_and_vars (while normalizing by policy_accum_steps)
        grads_and_vars = []
        for g, v in zip(self.accum_grads.values(), variables):
            grads_and_vars.append([tf.div(g, self.solver_params['n_accum_steps']), v])

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, accum_grads_op

    def train(self, objective, mode):
        setattr(self, 'loss_' + mode, objective)
        backward = self.backward(getattr(self, 'loss_' + mode))
        setattr(self, 'apply_grads_' + mode, backward[0])
        setattr(self, 'accum_grads_' + mode, backward[1])

    def create_variables(self):
        weights = OrderedDict([
            ('w0', tf.Variable(tf.random_normal([self.arch_params['in_dim'], self.arch_params['n_hidden_0']],
                                                stddev=self.solver_params['weights_stddev']))),
            ('w1', tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']],
                                                stddev=self.solver_params['weights_stddev']))),
            ('wc', tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']],
                                                stddev=self.solver_params['weights_stddev']))),
        ])

        biases = OrderedDict([
            ('b0', tf.Variable(
                tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev']))),
            ('b1', tf.Variable(
                tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev']))),
            ('bc',
             tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'])))
        ])
        return weights, biases

    def _init_layers(self):
        self.weights, self.biases = self.create_variables()

        weights, biases = self.create_variables()
        self.accum_grads = weights.copy()
        self.accum_grads.update(biases)

        self.reset_grad_op = []
        for acc_grad in self.accum_grads.values():
            self.reset_grad_op.append(acc_grad.assign(0. * acc_grad))
