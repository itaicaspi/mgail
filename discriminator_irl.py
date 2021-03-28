import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ops


class DiscriminatorIRL(object):
    def __init__(self, in_dim, out_dim, size, lr, do_keep_prob, weight_decay, state_only, gamma, state_size, action_size):
        self.arch_params = {
            'state_size': state_size,
            'action_size': action_size, 
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

        self.state_only = state_only
        self.gamma = gamma

    def forward(self, state, action, nstate, lprobs, reuse=False):

        with tf.variable_scope('discriminator'):
            # REWARD FN:

            rew_input = state
            rew_in_dim = self.arch_params['state_size']
            if not self.state_only:
                rew_in_dim = self.arch_params['in_dim']
                rew_input = tf.concat([state, action], axis=1)
            
            with tf.variable_scope('reward'): # hidden layers in AIRL code is 32, 32
                h0_rew = ops.dense(rew_input, rew_in_dim, self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0_rew', reuse)
                h1_rew = ops.dense(h0_rew, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1_rew', reuse)
                relu1_do_rew = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
                self.reward = ops.dense(relu1_do_rew, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2_rew', reuse)
            # END REWARD FN:

            # VALUE FN: WE NEED NEXT STATES!
            with tf.variable_scope('vfn'):
                # fitted_value_fn_n = value_fn_arch(self.nobs_t, dout=1) # THIS NEEDS TO CHANGE!!
                h0_nval = ops.dense(nstate, self.arch_params['state_size'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0_val', reuse)
                h1_nval = ops.dense(h0_nval, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1_val', reuse)
                relu1_do_nval = tf.nn.dropout(h1_nval, self.arch_params['do_keep_prob'])
                fitted_value_fn_n = ops.dense(relu1_do_nval, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2_val', reuse)
            
            with tf.variable_scope('vfn', reuse=True):
                # self.value_fn = fitted_value_fn = value_fn_arch(state, dout=1) 
                h0_val = ops.dense(state, self.arch_params['state_size'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0_val', reuse=True)
                h1_val = ops.dense(h0_val, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1_val', reuse=True)
                relu1_do_val = tf.nn.dropout(h1_val, self.arch_params['do_keep_prob'])
                self.value_fn = fitted_value_fn = ops.dense(relu1_do_val, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2_val', reuse=True)
            #  END VALUE FN

            self.qfn = self.reward + self.gamma*fitted_value_fn_n
            log_p_tau = self.reward  + self.gamma*fitted_value_fn_n - fitted_value_fn

            log_q_tau = lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            discrim_output = tf.exp(log_p_tau-log_pq)

            # concat = tf.concat(axis=1, values=[state, action])
            # h0 = ops.dense(concat, self.arch_params['in_dim'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0', reuse)
            # h1 = ops.dense(h0, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1', reuse)
            # relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
            # d = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)

        return discrim_output, log_p_tau, log_q_tau, log_pq

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

    
    def relu_net(x, layers=2, dout=1, d_hidden=32):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d'%i)
    out = linear(out, dout=dout, name='lfinal')
    return out