import pygetwindow as gw
import time

from Screen import Screen
from Cards import Cards
from AI import AI
from ClashRoyaleEnv import ClashRoyaleEnv
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

# Search for any windows open with Clash Royale in the title
game_window_list = gw.getWindowsWithTitle('Clash Royale')
game_window = []

# Exit if no windows are found
if not game_window_list:
    print("No game window found, please launch the game")
    exit()

# gw.getWindowsWithTitle does not match the string exactly so multiple windows could be fetched
for window in game_window_list:
    if window.title == "Clash Royale":
        game_window = window
        break

# Exit if no game window is found
if not game_window:
    print("No game window found, please launch the game")
    exit()

# Maximize the window and activate it to bring it to the front
try:
    if game_window.isMinimized:
        game_window.restore()
    game_window.maximize()
    game_window.activate()
except:
    pass

game_window_region = (game_window.left, game_window.top, (game_window.left + game_window.width), (game_window.top + game_window.height))
screen = Screen(game_window_region)
cards = Cards()
ai = AI()
env = ClashRoyaleEnv(ai, cards, screen)

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
model.learn(total_timesteps=100000, callback=callback)  # Train for 100k steps

# model = PPO.load("./train/best_model_100000", env=env)

# # Run a Trained AI Match
# obs, _ = env.reset()
# terminated = False
# total_reward = 0
# while not terminated:
#     action, _states = model.predict(obs)
#     obs, reward, terminated, truncated, info = env.step(action)
#     total_reward += reward
#     print(f"Action: {action}, Reward: {reward}")

# # Wait until final screen animation finishes
# time.sleep(8)

# # Check which player won
# winner = screen.game_winner_check()
# if winner:
#     print("Player won the game")
#     total_reward += 10000

# else:
#     print("Player lost the game")
#     total_reward -= 10000

# print(f"Total reward: {total_reward}")