import pygetwindow as gw
from stable_baselines3 import PPO

from Screen import Screen
from Cards import Cards
from AI import AI
from LoggingCallback import LoggingCallback
from ClashRoyaleEnv import ClashRoyaleEnv

def activate_game_window():
    # Maximize the window and activate it to bring it to the front
    try:
        if game_window.isMinimized:
            game_window.restore()
        game_window.maximize()
        game_window.activate()
    except:
        pass

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

activate_game_window()

# Initialise all classes
game_window_region = (game_window.left, game_window.top, (game_window.left + game_window.width), (game_window.top + game_window.height))
screen = Screen(game_window_region)
cards = Cards()
ai = AI()
env = ClashRoyaleEnv(ai, cards, screen)
callback = LoggingCallback(check_freq=5000)

user_input = input("Are you training or testing? (train/test) \n")

if user_input == "train":
    # Code for training the AI
    log_dir = './logs/'
    train_from = input("Are you training from scratch or a save state? (scratch/save) \n")

    if train_from == "scratch":
        # Start training from scratch
        activate_game_window()
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    elif train_from == "save":
        # Start training from desired save state
        model_start = int(input("What model number do you want to train from? \n best_model_"))
        activate_game_window()
        model = PPO.load("./train/best_model_{}".format(model_start), env=env)
        model.learn(total_timesteps=100000, callback=callback)

elif user_input == "test":
    # Code for testing the AI
    model_test = int(input("What model number do you want to test from? \n best_model_"))
    model = PPO.load("./train/best_model_{}".format(model_test), env=env)
    activate_game_window()

    games_played = 0
    games_won = 0

    # Run Trained AI Matches
    for i in range(100):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _states = model.predict(obs)
            # action = model.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}")

        # Check which player won
        winner = screen.game_winner_check()
        if winner:
            print("Player won the game")
            games_won += 1
        
        else:
            print("Player lost the game")
            
        games_played += 1
        print("Games won: {}".format(games_won))
        print("Games played: {}".format(games_played))

win_percentage = (games_won / games_played) * 100

print("Win percentage was: {}%".format(win_percentage))