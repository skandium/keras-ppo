"""
Simple Proximal Policy Optimization in Keras

"""
import gym
import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as keras_backend
from keras import optimizers

GAME = "CartPole-v0"

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
EPISODES = 1000
GAMMA = 0.99
BUFFER_SIZE = 256
BATCH_SIZE = 64
LR = 1e-4  # Lower lr stabilises training greatly
HIDDEN_DIMS = [64, 64]
ENTROPY_LOSS = 1e-3


class PPOAgent(object):

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_actor()
        self.__build_critic()

    def __build_actor(self):
        """Policy-based network that will make use of the PPO loss function"""
        advantage = layers.Input(shape=(1,))
        old_prediction = layers.Input(shape=(self.output_dim,))

        self.X = layers.Input(shape=(self.input_dim,))
        net = self.X

        for h_dim in HIDDEN_DIMS:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(self.output_dim)(net)
        net = layers.Activation("softmax")(net)

        model = Model(inputs=[self.X, advantage, old_prediction], outputs=[net])
        model.compile(optimizer=optimizers.Adam(lr=LR), loss=self.ppo_loss(advantage=advantage,
                                                                           old_prediction=old_prediction))
        model.summary()

        self.actor = model

    def __build_critic(self):
        """Value-based network that predicts the reward of each state"""

        inp = layers.Input(shape=(self.input_dim,))
        c_net = inp

        for h_dim in HIDDEN_DIMS:
            c_net = layers.Dense(h_dim)(c_net)
            c_net = layers.Activation("relu")(c_net)

        c_net = layers.Dense(1)(c_net)

        model = Model(inputs=inp, outputs=c_net)
        model.compile(optimizer=optimizers.Adam(lr=LR), loss="mse")
        model.summary()

        self.critic = model

    @staticmethod
    def ppo_loss(advantage, old_prediction):
        """Defined in https://arxiv.org/abs/1707.06347"""

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -keras_backend.mean(
                keras_backend.minimum(r * advantage, keras_backend.clip(r, min_value=1 - LOSS_CLIPPING,
                                                                        max_value=1 + LOSS_CLIPPING) * advantage) +
                ENTROPY_LOSS * -(
                        prob * keras_backend.log(prob + 1e-10)))

        return loss

    @staticmethod
    def discount_rewards(rewards, discount_rate=.99):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * discount_rate + rewards[t]
            discounted_r[t] = running_add
        discounted_r -= discounted_r.mean() / discounted_r.std()
        return discounted_r

    def act(self, state):
        state = np.expand_dims(state, axis=0)

        action_prob = np.squeeze(self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.output_dim))]))
        action = np.random.choice(np.arange(self.output_dim), p=action_prob)
        action_matrix = np.zeros(self.output_dim)
        action_matrix[action] = 1
        return action, action_matrix, action_prob

    def get_batch(self, env):
        s = env.reset()
        total_reward = []
        trials = [[], [], []]

        states = []
        actions = []
        predicted_actions = []
        rewards = []

        undiscounted_rewards = []

        while len(states) < BUFFER_SIZE:
            # env.render()
            action, action_matrix, predicted_action = self.act(state=s)
            observation, reward, done, info = env.step(action)
            total_reward.append(reward)

            trials[0].append(s)
            trials[1].append(action_matrix)
            trials[2].append(predicted_action)

            s = observation

            if done:
                undiscounted_rewards.append(sum(total_reward))
                discounted_reward = self.discount_rewards(rewards=total_reward, discount_rate=GAMMA)

                states.extend(trials[0])
                actions.extend(trials[1])
                predicted_actions.extend(trials[2])
                rewards.extend(discounted_reward)

                total_reward = []
                trials = [[], [], []]
                env.reset()

        obs, action, pred, reward = np.array(states), np.array(actions), np.array(predicted_actions), np.reshape(
            np.array(rewards), (len(rewards), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1]))
        return obs, action, pred, reward, undiscounted_rewards

    def run(self, env):
        episode = 0
        undiscounted_rewards = []
        while episode < EPISODES:
            print(episode)
            obs, action, pred, reward, undiscounted_reward = self.get_batch(env)
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[
                                                                                                     :BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values
            # advantage = (advantage - advantage.mean()) / advantage.std()

            self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True,
                           epochs=EPOCHS, verbose=False)
            self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                            verbose=False)
            undiscounted_rewards.extend(undiscounted_reward)
            if len(undiscounted_rewards) > 100:
                average_reward = np.mean(np.array(undiscounted_rewards)[-100:])
                print("Average reward over last 100 trials: ", average_reward)
            episode += 1


def main():
    env = gym.make(GAME)
    # supports discrete games for now
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = PPOAgent(input_dim, output_dim)
    agent.run(env)


if __name__ == '__main__':
    main()
