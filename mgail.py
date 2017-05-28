import numpy as np
import tensorflow as tf

import os
import common
from ER import ER
from forward_model import ForwardModel
from discriminator import Discriminator
from policy import Policy


class MGAIL(object):
    def __init__(self, environment):

        self.env = environment

        # Create placeholders for all the inputs
        self.states_ = tf.placeholder("float", shape=(None, self.env.state_size), name='states_')  # Batch x State
        self.states = tf.placeholder("float", shape=(None, self.env.state_size), name='states')  # Batch x State
        self.actions = tf.placeholder("float", shape=(None, self.env.action_size), name='action')  # Batch x Action
        self.label = tf.placeholder("float", shape=(None, 1), name='label')
        self.gamma = tf.placeholder("float", shape=(), name='gamma')
        self.temp = tf.placeholder("float", shape=(), name='temperature')
        self.noise = tf.placeholder("float", shape=(), name='noise_flag')
        self.do_keep_prob = tf.placeholder("float", shape=(), name='do_keep_prob')

        # Create MGAIL blocks
        self.forward_model = ForwardModel(state_size=self.env.state_size,
                                          action_size=self.env.action_size,
                                          encoding_size=self.env.fm_size,
                                          lr=self.env.fm_lr)

        self.discriminator = Discriminator(in_dim=self.env.state_size + self.env.action_size,
                                           out_dim=2,
                                           size=self.env.d_size,
                                           lr=self.env.d_lr,
                                           do_keep_prob=self.do_keep_prob,
                                           weight_decay=self.env.weight_decay)

        self.policy = Policy(in_dim=self.env.state_size,
                              out_dim=self.env.action_size,
                              size=self.env.p_size,
                              lr=self.env.p_lr,
                              do_keep_prob=self.do_keep_prob,
                              n_accum_steps=self.env.policy_accum_steps,
                              weight_decay=self.env.weight_decay)

        # Create experience buffers
        self.er_agent = ER(memory_size=self.env.er_agent_size,
                           state_dim=self.env.state_size,
                           action_dim=self.env.action_size,
                           reward_dim=1,  # stub connection
                           qpos_dim=self.env.qpos_size,
                           qvel_dim=self.env.qvel_size,
                           batch_size=self.env.batch_size,
                           history_length=1)

        self.er_expert = common.load_er(fname=os.path.join(self.env.run_dir, self.env.expert_data),
                                        batch_size=self.env.batch_size,
                                        history_length=1,
                                        traj_length=2)

        self.env.sigma = self.er_expert.actions_std / self.env.noise_intensity

        # Normalize the inputs
        states_ = common.normalize(self.states_, self.er_expert.states_mean, self.er_expert.states_std)
        states = common.normalize(self.states, self.er_expert.states_mean, self.er_expert.states_std)
        if self.env.continuous_actions:
            actions = common.normalize(self.actions, self.er_expert.actions_mean, self.er_expert.actions_std)
        else:
            actions = self.actions

        # 1. Forward Model
        initial_gru_state = np.ones((1, self.forward_model.encoding_size))
        forward_model_prediction, _ = self.forward_model.forward([states_, actions, initial_gru_state])
        forward_model_loss = tf.reduce_mean(tf.square(states-forward_model_prediction))
        self.forward_model.train(objective=forward_model_loss)

        # 2. Discriminator
        labels = tf.concat([1 - self.label, self.label], 1)
        d = self.discriminator.forward(states, actions)

        # 2.1 0-1 accuracy
        correct_predictions = tf.equal(tf.argmax(d, 1), tf.argmax(labels, 1))
        self.discriminator.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        # 2.2 prediction
        d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
        # cost sensitive weighting (weight true=expert, predict=agent mistakes)
        d_loss_weighted = self.env.cost_sensitive_weight * tf.multiply(tf.to_float(tf.equal(tf.squeeze(self.label), 1.)), d_cross_entropy) +\
                                                           tf.multiply(tf.to_float(tf.equal(tf.squeeze(self.label), 0.)), d_cross_entropy)
        discriminator_loss = tf.reduce_mean(d_loss_weighted)
        self.discriminator.train(objective=discriminator_loss)

        # 3. Collect experience
        mu = self.policy.forward(states)
        if self.env.continuous_actions:
            a = common.denormalize(mu, self.er_expert.actions_mean, self.er_expert.actions_std)
            eta = tf.random_normal(shape=tf.shape(a), stddev=self.env.sigma)
            self.action_test = tf.squeeze(a + self.noise * eta)
        else:
            a = common.gumbel_softmax(logits=mu, temperature=self.temp)
            self.action_test = tf.argmax(a, dimension=1)

        # 4.3 AL
        def policy_loop(state_, t, total_cost, total_trans_err, _):
            mu = self.policy.forward(state_, reuse=True)

            if self.env.continuous_actions:
                eta = self.env.sigma * tf.random_normal(shape=tf.shape(mu))
                action = mu + eta
            else:
                action = common.gumbel_softmax_sample(logits=mu, temperature=self.temp)

            # minimize the gap between agent logit (d[:,0]) and expert logit (d[:,1])
            d = self.discriminator.forward(state_, action, reuse=True)
            cost = self.al_loss(d)

            # add step cost
            total_cost += tf.multiply(tf.pow(self.gamma, t), cost)

            # get action
            if self.env.continuous_actions:
                a_sim = common.denormalize(action, self.er_expert.actions_mean, self.er_expert.actions_std)
            else:
                a_sim = tf.argmax(action, dimension=1)

            # get next state
            state_env, _, env_term_sig, = self.env.step(a_sim, mode='tensorflow')[:3]
            state_e = common.normalize(state_env, self.er_expert.states_mean, self.er_expert.states_std)
            state_e = tf.stop_gradient(state_e)

            state_a, _ = self.forward_model.forward([state_, action, initial_gru_state], reuse=True)

            state, nu = common.re_parametrization(state_e=state_e, state_a=state_a)
            total_trans_err += tf.reduce_mean(abs(nu))
            t += 1

            return state, t, total_cost, total_trans_err, env_term_sig

        def policy_stop_condition(state_, t, cost, trans_err, env_term_sig):
            cond = tf.logical_not(env_term_sig)
            cond = tf.logical_and(cond, t < self.env.n_steps_train)
            cond = tf.logical_and(cond, trans_err < self.env.total_trans_err_allowed)
            return cond

        state_0 = tf.slice(states, [0, 0], [1, -1])
        loop_outputs = tf.while_loop(policy_stop_condition, policy_loop, [state_0, 0., 0., 0., False])
        self.policy.train(objective=loop_outputs[2])

    def al_loss(self, d):
        logit_agent, logit_expert = tf.split(axis=1, num_or_size_splits=2, value=d)

        # Cross entropy loss
        labels = tf.concat([tf.zeros_like(logit_agent), tf.ones_like(logit_expert)], 1)
        d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
        loss = tf.reduce_mean(d_cross_entropy)

        return loss*self.env.policy_al_w
