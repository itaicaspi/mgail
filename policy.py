from collections import OrderedDict
import tensorflow as tf
import ops
import copy

class Policy(object):
    def __init__(self, in_dim, out_dim, size, lr, do_keep_prob, n_accum_steps, weight_decay):

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
            'n_accum_steps': n_accum_steps,
        }
        self._init_layers()
        self.create_variables()

    def forward(self, state):
        h0 = tf.nn.relu(tf.matmul(state, self.weights["dense0_weights"]) + self.weights["dense0_biases"])
        h1 = tf.nn.relu(tf.matmul(h0, self.weights["dense1_weights"]) + self.weights["dense1_biases"])
        relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
        a = tf.matmul(relu1_do, self.weights["dense2_weights"]) + self.weights["dense2_biases"]

        return a

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for k, v in self.weights.items() if 'weights' in k])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values())

        # get clipped gradients
        grads = [tf.clip_by_value(g, -2, 2) for g, v in grads_and_vars]

        variables = [v for g, v in grads_and_vars]

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

    def train(self, objective):
        self.loss_al = objective
        self.apply_grads_al, self.accum_grads_al = self.backward(self.loss_al)

    def create_variables(self):
        # we create all the weights and biases once and reuse them between graph runs
        self.weights = OrderedDict()
        self.weights.update(ops.linear_variables(self.arch_params['in_dim'], self.arch_params['n_hidden_0'], 'dense0'))
        self.weights.update(ops.linear_variables(self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], 'dense1'))
        self.weights.update(ops.linear_variables(self.arch_params['n_hidden_1'], self.arch_params['out_dim'], 'dense2'))

    def _init_layers(self):
        self.create_variables()
        self.accum_grads = self.weights.copy()

        self.reset_grad_op = []
        for acc_grad in self.accum_grads.values():
            self.reset_grad_op.append(acc_grad.assign(0. * acc_grad))
