import tensorflow as tf
import ops


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

    def forward(self, state, reuse=False):
        with tf.variable_scope('policy'):
            h0 = ops.dense(state, self.arch_params['in_dim'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0', reuse)
            h1 = ops.dense(h0, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1', reuse)
            relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
            a = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)

        return a

    def backward(self, loss):

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')

        self.accum_grads = [tf.Variable(tf.zeros(w.get_shape())) for w in self.weights]

        # reset gradients op
        self.reset_grad_op = []
        for acc_grad in self.accum_grads:
            self.reset_grad_op.append(acc_grad.assign(0. * acc_grad))

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(w) for w in self.weights if 'weights' in w.name])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights)

        # get clipped gradients
        grads = [tf.clip_by_value(g, -2, 2) for g, v in grads_and_vars]

        variables = [v for g, v in grads_and_vars]

        # accumulate the grads
        accum_grads_op = []
        for i, accum_grad in enumerate(self.accum_grads):
            accum_grads_op.append(accum_grad.assign_add(grads[i]))

        # pack accumulated gradient and vars back in grads_and_vars (while normalizing by policy_accum_steps)
        grads_and_vars = []
        for g, v in zip(self.accum_grads, variables):
            grads_and_vars.append([tf.div(g, self.solver_params['n_accum_steps']), v])

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, accum_grads_op

    def train(self, objective):
        self.loss_al = objective
        self.apply_grads_al, self.accum_grads_al = self.backward(self.loss_al)
