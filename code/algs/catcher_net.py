import tensorflow as tf
from tensorflow.keras import layers

class CatcherEncoder(tf.keras.Model):
    def __init__(self, state_dim):
        super().__init__()
        self.lstm1 = layers.LSTM(64, return_sequences=True)
        self.lstm2 = layers.LSTM(64)
        self.dense_after_lstm = layers.Dense(128, activation='relu')

        self.feature_extractor = tf.keras.Sequential([
            layers.Reshape((state_dim,)),  # reshape to flat vector
            layers.Dense(128, activation='relu')
        ])

    def call(self, state):
        lstm_out = self.lstm1(state)
        lstm_out = self.lstm2(lstm_out)
        lstm_proj = self.dense_after_lstm(lstm_out)

        feature = self.feature_extractor(tf.reshape(state, (state.shape[0], -1))) 
        return tf.concat([lstm_proj, feature], axis=-1)


class CatcherActor(tf.keras.Model):
    def __init__(self, state_dim):
        super().__init__()
        self.encoder = CatcherEncoder(state_dim)
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='tanh')  # [-1, 1]

    def call(self, state):
        x = self.encoder(state)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output_layer(x)

class CatcherCritic(tf.keras.Model):
    def __init__(self, state_dim):
        super().__init__()
        self.encoder = CatcherEncoder(state_dim)
        self.concat = layers.Concatenate()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation=None)  # Q value

    def call(self, state, action):
        x = self.encoder(state)
        x = self.concat([x, action])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output_layer(x)
