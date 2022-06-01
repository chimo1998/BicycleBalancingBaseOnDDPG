from numpy.core.fromnumeric import nonzero, transpose
from six import b
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import socket

import os
import pandas as pd

import time

from tensorflow.python.keras.backend import batch_flatten
from tensorflow.keras.utils import plot_model

# Classes

class Buffer:
    def __init__(self, buffer_capacity=30000, batch_size=128):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_count = 0

        self.state_buffer = np.zeros((self.buffer_capacity, NUM_STATES))
        self.action_buffer = np.zeros((self.buffer_capacity, NUM_ACTIONS))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, NUM_STATES))

    def record(self, obs_tuple):
        index = self.buffer_count % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_count += 1

    def save(self):
        saf = pd.DataFrame({"state": self.state_buffer.tolist(), "action": self.action_buffer.tolist(),
                                        "reward": self.reward_buffer.tolist(), "next": self.next_state_buffer.tolist()})
        # saf = pd.DataFrame.from_dict({"angel":columns[0], "angular_velocity":columns[1], "angular_acc":columns[2], "action":action_list})
        saf.to_csv("buffer.csv",
                    index=False, encoding="utf-8")

        saf = pd.DataFrame({'size': [self.buffer_count]})
        saf.to_csv('buffer_size.csv', index=False, encoding='utf-8')

    def load(self):
        df = pd.read_csv('buffer.csv')
        sb = df['state'].to_numpy()
        ssb = [s.replace('[', '').replace(']','').split(', ') for s in sb]
        self.state_buffer = np.array(ssb, np.float32)

        sb = df['action'].to_numpy()
        self.action_buffer = np.array([s.replace('[', '').replace(']','').split(', ') for s in sb], np.float32)
        sb = df['reward'].to_numpy()
        self.reward_buffer = np.array([s.replace('[', '').replace(']','').split(', ') for s in sb], np.float32)

        sb = df['next'].to_numpy()
        ssb = [s.replace('[', '').replace(']','').split(', ') for s in sb]
        self.next_state_buffer = np.array(ssb, np.float32)

        df = pd.read_csv('buffer_size.csv')
        self.buffer_count = int(df['size'][0])
        print(self.buffer_count)

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            # Use target actor_model predict target actions
            target_actions = target_actor(next_state_batch, training=True)
            # Target y
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            # Q
            critic_value = critic_model(
                [state_batch, action_batch], training=True)
            # Critic loss = MSE of (target - Q)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # Gradient decent
        critic_grad = tape.gradient(
            critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Use `-value` as we want to maximize the value
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self):
        # Sample and random
        record_range = min(self.buffer_count, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        print(type(batch_indices))

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.05, dt=1e-2, x_iniitial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_iniitial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# Methods


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    if training:
        noise = np.clip(noise_object(), -1, 1) * (max(0.0, 1-(max(0,ep-100)/800.0)))
        # print("noise %f" % noise)
        sampled_actions = (sampled_actions.numpy() + noise)
        # sampled_actions = (sampled_actions.numpy() + noise + np.random.normal(scale=0.3))
        # print(sampled_actions)

    legal_action = np.clip(sampled_actions, LOWER_BOUND, UPPER_BOUND)
    action_list.append(legal_action)

    return [np.squeeze(legal_action)]


actoooooor = True
criticcccc = True


def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

    inputs = layers.Input(shape=(NUM_STATES,))
    out = layers.Dense(128, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh",
                           kernel_initializer=last_init)(out)

    outputs = outputs * UPPER_BOUND
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    state_input = layers.Input(shape=(NUM_STATES))
    state_out = layers.Dense(64, activation="relu")(state_input)
    state_out = layers.Dense(96, activation="relu")(state_out)

    action_inputs = layers.Input(shape=(NUM_ACTIONS))
    action_out = layers.Dense(64, activation="relu")(action_inputs)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="softmax")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_inputs], outputs)

    return model


def get_state(state):
    state_list.append(state)

    global last_reward

    # state, reward, done, info
    done = False

    reward_stand = (1*(state[0])**2 + 0.01*state[1]**2 + (0.005*state[2]**2))
    # reward_compitition = (((1+state[3]+state[4])) / (1+abs(state[0])))
    # reward = -(alpha*reward_stand + (1-alpha)*reward_compitition)
    reward = - reward_stand

    if np.abs(state[0]) > 15:
        print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
        reward = last_reward
        done = True

    # state = [state[i] / div[i] for i in range(3)]

    last_reward = reward
    return state, reward, done

# Main


NUM_STATES = 5
NUM_ACTIONS = 1

fps = 40
sleep_time = (float)(1/fps)

UPPER_BOUND = 5.0
LOWER_BOUND = -5.0

last_reward = 0
alpha = 1.0
alpha_max = 0.8
alpha_min = 0.5

std_dev = 0.3
ou_noise = OUActionNoise(mean=np.zeros(
    1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.004
actor_lr = 0.002

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 1500
episode_time = 30
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(80000, 64)

ep_reward_list = []
avg_reward_list = []
time_list = []
ep_list = []
state_list = []
action_list = []

print()
print()
print()
print()
print()
print("Waiting for connection")

client = None

STATE_TRAINING = 1
STATE_RUNNING = 2
STATE_TESTING = 3
testing_episodes = 21
state = 1
training = state == STATE_TRAINING
#weights = "20211202/h5/actor_model-1200.h5"
#model_name = "20220411_30s/h5/actor_model-640"
# model_name = "20220411_r/h5/actor_model-540"

prev_state=None
reward=0
done=None
action=0
episodic_reward = 0
base_on = False
keep = False

def init(actor, critic, var_base_on, var_keep):
    global ep, actor_model, critic_model, base_on, keep
    base_on = var_base_on
    keep = var_keep
    if ( keep ):
        ep = int(len(os.listdir('./h5'))/2-1)
        print('./h5/actor_model-%da'%ep)
        # actor_model = tf.saved_model.load('./h5/actor_model-%d'%ep, tags='None')
        actor_model = keras.models.load_model('h5/actor_model-%d'%ep, compile=False)
        target_actor = actor_model
        # critic_model = tf.saved_model.load('./h5/critic_model-%d'%ep)
        critic_model = keras.models.load_model('h5/critic_model-%d'%ep, compile=False)
        target_critic = critic_model
    elif ( base_on ):
        actor_model = tf.keras.models.load_model(actor, compile=False)
        target_actor = actor_model
        critic_model = tf.keras.models.load_model(critic, compile=False)
        target_critic = critic_model

    global buffer, ep_reward_list, avg_reward_list, time_list, ep_list, state_list, action_list, episodic_reward
    episodic_reward = 0
    buffer = Buffer(80000, 64)
    if (keep or base_on):
        buffer.load()
    ep_reward_list = []
    avg_reward_list = []
    time_list = []
    ep_list = []
    state_list = []
    action_list = []
    global prev_state, reward, done
    prev_state, reward, done = get_state((0,0,0,0,0))
    global tt
    tt = time.time()

def episode_start():
    global ep_reward_list, state_list, action_list, episodic_reward
    episodic_reward = 0
    action_list = []
    global prev_state, reward, done
    prev_state, reward, done = get_state((0,0,0,0,0))
    state_list = []
    global tt, ep
    tt = time.time()
    ep+=1


ep = 0

prev_reward = 0
prev_action = 0
prev_done = False
tt = time.time()
def predict(input_state):
    global prev_state, done, alpha, ep, tt
    global reward, action
    global buffer, ep_reward_list, avg_reward_list, time_list, ep_list, state_list, action_list, episodic_reward

    alpha = alpha_max+(alpha_min-alpha_max)*float(min(1.0, (ep/300.0)))

    buffer.record((prev_state, action, reward, input_state))
    buffer.learn()
    update_target(target_actor.variables, actor_model.variables, tau)
    update_target(target_critic.variables, critic_model.variables, tau)


    state, reward, done = get_state(input_state)
    episodic_reward += reward

    if done:
        return None

    tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

    action = policy(tf_state, ou_noise)


    prev_state = state
    return action[0]

def save():
    global state_list, ep_list, time_list, avg_reward, ep_reward_list, avg_reward_list, action_list, state_list
    if len(state_list) == 0:
        return
    ep_reward_list.append(episodic_reward)
    ep_list.append(ep)
    time_list.append(time.time()-tt)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    actor_model.save("h5/actor_model-%d"%ep)
    critic_model.save("h5/critic_model-%d"%ep)
    try:
        df = pd.DataFrame.from_dict(
            {"ep": ep_list, "ep_reward": ep_reward_list, "time": time_list})
    except:
        return
    df.to_csv("history/his-%d.csv"%(ep),
                index=False, encoding='utf-8')

    action_list.append(999)
    if(len(state_list) != len(action_list)):
        state_list.append(state_list[-1])

    columns = list(zip(*state_list))
    try:
        saf = pd.DataFrame.from_dict({"angel": columns[0], "angular_velocity": columns[1],
                                        "angular_acc": columns[2], "motor1": columns[3], "motor2": columns[4], "action": action_list})
        # saf = pd.DataFrame.from_dict({"angel":columns[0], "angular_velocity":columns[1], "angular_acc":columns[2], "action":action_list})
        saf.to_csv("states/sa-%d.csv"%(ep),
                    index=False, encoding="utf-8")
    except Exception as e:
        print(len(action_list))
        print(len(columns[0]))
        print(e)
        return
    buffer.save()
    state_list = []
    action_list = []