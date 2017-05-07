import cPickle
import math
import time
import tensorflow as tf
import numpy as np
import os
import sys

def get_params(obj):
    params = {}
    for param in obj:
        params[param.name] = param.get_value()
    return params


def dotproduct(v1, v2):
    return sum((a*b) for a,b in zip(v1,v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def save_params(fname, saver, session):
    saver.save(session, fname)


def load_params(fName):
    f = file(fName,'rb')
    obj = cPickle.load(f)
    f.close()
    return obj


def create_lr_func(solver_params):
    if solver_params['lr_type'] == 'inv':
        return inv_learning_rate
    elif solver_params['lr_type'] == 'fixed':
        return fixed_learning_rate
    elif solver_params['lr_type'] == 'episodic':
        return episodic_learning_rate
    else:
        return []


def inv_learning_rate(itr, solver_params):
    return solver_params['base'] * (1 + solver_params['gamma'] * itr) ** (-solver_params['power'])


def fixed_learning_rate(itr, solver_params):
    return solver_params['base']


def episodic_learning_rate(itr, solver_params):
    return solver_params['base'] / (math.floor(itr / solver_params['interval']) + 1)


def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g, w in grads_and_vars:
        tot_grad += tf.reduce_mean(tf.abs(g))
        tot_w += tf.reduce_mean(tf.abs(w))

    return tot_grad/N, tot_w/N


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def uniform_initializer(shape):
    scale = np.sqrt(6. / sum(get_fans(shape)))
    weight = (np.random.uniform(-1, 1, size=shape) * scale).astype(np.float32)
    return weight


def print_tensor_stat(x, name):
    x_mean = tf.reduce_mean(x)
    x_std = tf.reduce_mean(tf.square(x - x_mean))
    x_max = tf.reduce_max(x)
    x = tf.Print(x, [x_mean, x_std, x_max], message=name)
    return x


def calculate_gamma(itrvl, gamma_f, t):
    return np.clip(0.1 * int(t/itrvl), 0, gamma_f)


def multivariate_pdf_tf(x, mu, sigma):
    inv_sqrt_2pi = (1. / np.sqrt(2 * np.pi)).astype(np.float32)
    A = tf.mul(inv_sqrt_2pi, tf.div(1., sigma))
    B = tf.reduce_sum(tf.mul(tf.square(x - mu), sigma), reduction_indices=[1])
    p_x = tf.mul(A, tf.exp(tf.mul(-0.5, B)))
    p_x = tf.stop_gradient(p_x)
    return p_x


def multivariate_pdf_np(x, mu, sigma):
    inv_sqrt_2pi = (1. / np.sqrt(2 * np.pi)).astype(np.float32)
    A = inv_sqrt_2pi / sigma
    B = ((x - mu) ** 2 * sigma).sum()
    p_x = A * np.exp(-0.5 * B)
    return p_x


def save_er(module, directory, exit_=False):
    fname = directory + '/er' + time.strftime("-%Y-%m-%d-%H-%M") + '.bin'
    f = file(fname, 'wb')
    cPickle.dump(module, f)
    print 'saved ER: %s' % fname
    if exit_:
        sys.exit(0)


def load_er(fname, batch_size, history_length, traj_length):
    f = file(fname, 'rb')
    er = cPickle.load(f)
    er.batch_size = batch_size
    er = set_er_stats(er, history_length, traj_length)
    return er


def set_er_stats(er, history_length, traj_length):
    state_dim = er.states.shape[-1]
    action_dim = er.actions.shape[-1]
    er.prestates = np.empty((er.batch_size, history_length, state_dim), dtype=np.float32)
    er.poststates = np.empty((er.batch_size, history_length, state_dim), dtype=np.float32)
    er.traj_states = np.empty((er.batch_size, traj_length, state_dim), dtype=np.float32)
    er.traj_actions = np.empty((er.batch_size, traj_length-1, action_dim), dtype=np.float32)
    er.states_min = np.min(er.states[:er.count], axis=0)
    er.states_max = np.max(er.states[:er.count], axis=0)
    er.actions_min = np.min(er.actions[:er.count], axis=0)
    er.actions_max = np.max(er.actions[:er.count], axis=0)
    er.states_mean = np.mean(er.states[:er.count], axis=0)
    er.actions_mean = np.mean(er.actions[:er.count], axis=0)
    er.states_std = np.std(er.states[:er.count], axis=0)
    er.states_std[er.states_std == 0] = 1
    er.actions_std = np.std(er.actions[:er.count], axis=0)
    return er


def scalar_to_4D(x):
    return tf.expand_dims(tf.expand_dims(tf.expand_dims(x, -1), -1), -1)


def add_field(self, name, size):
    setattr(self, name + '_field', np.arange(self.f_ptr, self.f_ptr + size))
    self.f_ptr += size


def compile_modules(run_dir):
    cwd = os.getcwd()
    os.chdir(run_dir)
    os.system('g++ -std=c++11 simulator.c -o simulator')
    os.system('g++ -std=c++11 -shared pipe.cc -o pipe.so -fPIC -I $TF_INC')
    os.chdir(cwd)


def relu(x, alpha=1./5.5):
    return tf.maximum(alpha * x, x)


def re_parametrization(state_e, state_a):
    nu = state_e - state_a
    nu = tf.stop_gradient(nu)
    return state_a + nu, nu


def logfunc(x, x2):
    return tf.mul(x, tf.log(tf.div(x, x2)))


def kl_div(rho, rho_hat):
    invrho = tf.sub(tf.constant(1.), rho)
    invrhohat = tf.sub(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho


def normalize(x, mean, std):
    return (x - mean)/std


def denormalize(x, mean, std):
    return x * std + mean


def _choice(mu):
    ind = np.random.choice(a=mu.shape[0], p=mu)
    a = 0 * mu
    a[ind] = 1.
    return a


def choice(mu):
    x = tf.py_func(_choice, inp=[mu], Tout=[tf.float32], name='choice_func')
    x = tf.reshape(x, mu.get_shape())
    return x


def decimal_to_one_hot(x, width):
    x_one_hot = np.zeros((x.shape[0], width))
    for i, elem in enumerate(x):
        x_one_hot[i, elem] = 1
    return x_one_hot


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
