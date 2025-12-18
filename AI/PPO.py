import gymnasium as gym
from gymnasium import spaces
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([4, 10001])
        self.observation_space = spaces.Dict({
            "hand_cards": spaces.Box(0, 52, shape=(2,), dtype=int),
            "community_cards": spaces.Box(0, 52, shape=(5,), dtype=int),
            "player_chips": spaces.Discrete(10000),
            "opponent_chips": spaces.Discrete(10000),
        })
        self.reset()

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            random.seed(seed)

        self.player_chips = 1000
        self.opponent_chips = 1000
        self.hand_cards = self.deal_cards(2)
        self.community_cards = self.deal_cards(5)

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "hand_cards": self.hand_cards,
            "community_cards": self.community_cards,
            "player_chips": self.player_chips,
            "opponent_chips": self.opponent_chips,
        }

    def deal_cards(self, num):
        return random.sample(range(1, 53), num)

    def step(self, action):
        action_type = action[0]  # Fold, Call, Raise, Check
        raise_amount = action[1]

        reward = self._calculate_reward(action_type, raise_amount)
        done = self.player_chips <= 0 or self.opponent_chips <= 0

        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def _calculate_reward(self, action_type, raise_amount):
        if action_type == 0:  # Fold
            return -10
        elif action_type == 1:  # Call
            return random.randint(-50, 50)
        elif action_type == 2:  # Raise
            return random.randint(-100, 100) * raise_amount
        elif action_type == 3:  # Check
            return 0


def make_env():
    return PokerEnv()

env = make_vec_env(make_env, n_envs=4)

#  PPO 算法，并使用 MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("poker_ai")
