import numpy as np
import tensorflow as tf
from .catcher_net import CatcherActor, CatcherCritic

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU available and memory growth enabled.")
    except RuntimeError as e:
        print("GPU config error:", e)
else:
    print("No GPU detected. Training will use CPU.")

class ReplayBuffer:
    def __init__(self, size, state_shape):
        self.size = size
        self.ptr = 0
        self.count = 0
        self.states = np.zeros((size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, *state_shape), dtype=np.float32)

    def add(self, s, a, r, s2):
        a = np.array(a, dtype=np.float32).reshape(1)
        r = np.array(r, dtype=np.float32).reshape(1)
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)


    def sample(self, batch_size):
        idxs = np.random.choice(self.count, batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs])

class CatcherDDPG:
    def __init__(self, state_dim, actor_lr=1e-3, critic_lr=1e-3, tau=0.02, gamma=0.9):
        self.actor = CatcherActor(state_dim)
        self.critic = CatcherCritic(state_dim)
        self.target_actor = CatcherActor(state_dim)
        self.target_critic = CatcherCritic(state_dim)
        self.buffer = ReplayBuffer(10000, (state_dim, 1))

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.tau = tau
        self.gamma = gamma
        self.update_target(1.0)

    def update_target(self, tau=None):
        tau = tau or self.tau
        for a, ta in zip(self.actor.variables, self.target_actor.variables):
            ta.assign(tau * a + (1 - tau) * ta)
        for c, tc in zip(self.critic.variables, self.target_critic.variables):
            tc.assign(tau * c + (1 - tau) * tc)

    def act(self, state):
        state = tf.convert_to_tensor(state[None, :, :], dtype=tf.float32)
        action = self.actor(state)
        prob = (action + 1) / 2  # normalize to [0, 1]
        return prob.numpy()[0, 0]

    def train(self, batch_size=128):
        if self.buffer.count < batch_size:
            return
        s, a, r, s2 = self.buffer.sample(batch_size)
        s = tf.convert_to_tensor(s)
        a = tf.convert_to_tensor(a)
        r = tf.convert_to_tensor(r)
        s2 = tf.convert_to_tensor(s2)

        # Update critic
        with tf.GradientTape() as tape:
            target_q = self.target_critic(s2, self.target_actor(s2))
            y = r + self.gamma * target_q
            q = self.critic(s, a)
            loss_critic = tf.reduce_mean(tf.square(y - q))
        grads = tape.gradient(loss_critic, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            action = self.actor(s)
            loss_actor = -tf.reduce_mean(self.critic(s, action))
        grads = tape.gradient(loss_actor, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        self.update_target()
