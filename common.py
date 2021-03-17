import pickle
import tensorflow as tf
import numpy as np
import h5py

from ER import ER

def save_params(fname, saver, session):
    saver.save(session, fname)

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys

def get_d4rl_dataset(h5path):
    dataset_file = h5py.File(h5path, 'r')
    data_dict = {}
    for k in get_keys(dataset_file):
        try:
            # first try loading as an array
            data_dict[k] = dataset_file[k][:]
        except ValueError as e: # try loading as a scalar
            data_dict[k] = dataset_file[k][()]
    dataset_file.close()

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:,0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:,0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (str(data_dict['rewards'].shape))
    return data_dict

def load_d4rl_er(h5path, batch_size, history_length, traj_length):
    data_dict = get_d4rl_dataset(h5path)
    data_size = data_dict["rewards"].shape[0]
    flattened_states = data_dict["observations"].reshape(data_size, -1)
    flattened_post_states = np.roll(flattened_states, -1, axis=0)
    flattened_post_states[-1] = flattened_post_states[-2] # the last post-state uses the pre-state
    terminals = data_dict["terminals"]
    inverted_terminals= np.invert(terminals)
    # masked out other states, only keep terminal states
    terminal_post_states = np.ma.masked_array(
        flattened_states,
        mask=np.column_stack([inverted_terminals for _ in range(flattened_post_states.shape[-1])]),
        fill_value=0
    )
    # masked out terminal states
    flattened_post_states = np.ma.masked_array(
        flattened_post_states,
        mask=np.column_stack([terminals for _ in range(flattened_post_states.shape[-1])]),
        fill_value=0
    )
    # add back the terminal states
    flattened_post_states += terminal_post_states
    state_dim = flattened_states.shape[-1]
    er = ER(data_size, state_dim, np.shape(data_dict['actions'])[1], batch_size, history_length)
    er.add(data_dict["actions"], data_dict["rewards"], flattened_states, terminals)
    er = set_er_stats(er, history_length, traj_length)
    return er

def load_er(fname, batch_size, history_length, traj_length):
    f = open(fname, 'rb')
    er = pickle.load(f)
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


def re_parametrization(state_e, state_a):
    nu = state_e - state_a
    nu = tf.stop_gradient(nu)
    return state_a + nu, nu


def normalize(x, mean, std):
    return (x - mean)/std


def denormalize(x, mean, std):
    return x * std + mean


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.compat.v1.random_uniform(shape,minval=0,maxval=1)
    return -tf.compat.v1.log(-tf.compat.v1.log(U + eps) + eps)


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
        y_hard = tf.cast(tf.equal(y, tf.compat.v1.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
