import gymnasium as gym
import numpy as np
import time
from Screen import Screen
from AI import AI
from Cards import Cards

class ClashRoyaleEnv(gym.Env):
    '''
    Class that models the Clash Royale game environment
    '''
    def __init__(self, ai: AI, cards: Cards, screen: Screen):
        super(ClashRoyaleEnv, self).__init__()

        # Allow environment to interact with other classes
        self.ai = ai
        self.screen = screen
        self.cards = cards

        # Maximum amount of troops and towers allowed for each player
        max_troops = 20
        max_towers = 3

        # Get the current menu screen which can be used to get the current players deck
        menu_screen = self.screen.get_menu_screen()
        self.deck_info = self.screen.get_deck_info(menu_screen, self.cards.card_info)

        # A game lasts 180 seconds if it does not go to overtime
        self.game_time = 180
        # Overtime lasts 120 seconds if game is not decisive
        self.overtime_time = 120

        # Initially end times are 0
        self.end_time = 0
        self.overtime_end_time = 0

        # Initially we are not in overtime and game is not finished
        self.overtime = False
        self.game_over = False
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=6000, shape=(
                (4 * 13) +  # 4 cards in hand, each with 13 stats
                1 +       # elixir
                (max_troops * 13) +  # max ally troops, each with 13 stats
                (max_troops * 13) +  # max enemy troops, each with 13 stats
                max_towers +  # max number of ally towers
                max_towers +  # max number of enemy towers
                1  # time remaining
            ,), dtype=np.float32
        )

        # Action space is the choice of 5 options (nothing, card1, card2, card3, card4) with the x and y coordinates
        self.action_space = gym.spaces.MultiDiscrete([5, 510, 745])

        # Reward based on change in tower hp
        self.previous_min_ally_tower_hp = []
        self.previous_min_enemy_tower_hp = []
        self.previous_ally_towers_destroyed = []
        self.previous_enemy_towers_destroyed = []

    def get_observation(self):
        '''
        Gets the observations from the screen
        '''
        cards_in_hand = self.screen.get_cards_in_hand(self.deck_info)

        # Get the card stats of the players starting hand
        cards_in_hand_stats = []
        for card in cards_in_hand:
            cards_in_hand_stats.append(self.cards.get_card_stats(card))

        found_troops = self.screen.detect_troops()

        ally_troop_stats = []
        enemy_troop_stats = []

        for result in found_troops:
            # Extract x,y coordinates and troop name from results
            for i in range(0, len(result.boxes.cls)):
                x = round(result.boxes.xywh[i][0].item()) + round(result.boxes.xywh[i][2].item() / 2)
                y = round(result.boxes.xywh[i][1].item()) + round(result.boxes.xywh[i][3].item() / 2)
                troop_name = str(result.names[result.boxes.cls[i].item()])
                troop_type = troop_name.split("_")[0]

                if troop_type == "ally":
                    # Remove ally_ prefix
                    troop_name = troop_name[5:]

                    # Create troop_stats array and append to ally_troop_stats
                    troop_stats = self.cards.get_troop_stats(troop_name)
                    if troop_stats is None:
                        continue
                    else:
                        troop_stats.append(x)
                        troop_stats.append(y)
                        ally_troop_stats.append(troop_stats)


                elif troop_type == "enemy":
                    # Remove enemy_ prefix
                    troop_name = troop_name[6:]

                    # Create troop_stats array and append to enemy_troop_stats
                    troop_stats = self.cards.get_troop_stats(troop_name)
                    if troop_stats is None:
                        continue
                    else:
                        troop_stats.append(x)
                        troop_stats.append(y)
                        enemy_troop_stats.append(troop_stats)

        if len(ally_troop_stats) > 20:
            ally_troop_stats = ally_troop_stats[0:20]
        elif len(ally_troop_stats) < 20:
            padded_amount = 20 - len(ally_troop_stats)
            for i in range(padded_amount):
                ally_troop_stats.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Pad results with 0's or remove data if too large
        if len(enemy_troop_stats) > 20:
            enemy_troop_stats = enemy_troop_stats[0:20]
        elif len(enemy_troop_stats) < 20:
            padded_amount = 20 - len(enemy_troop_stats)
            for i in range(padded_amount):
                enemy_troop_stats.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


        # test tower hp detection
        ally_tower_hp, enemy_tower_hp = self.screen.get_tower_hp()

        # test elixir detection
        elixir = self.screen.get_elixir_count()

        if self.overtime == False:
            # time remaining in game
            time_remaining = self.end_time - time.time()
            # Check if we are in overtime
            if time_remaining <= 0:
                overtime = True

        else:
            # time remaining in overtime
            time_remaining = self.overtime_end_time - time.time()

        # Data to pass into AI: ally_troop_stats, ally_tower_hp, enemy_troop_stats, enemy_tower_hp, time_remaining, elixir, cards_in_hand_stats
        # Data output: integer 0-3 indicating card to select in current hand, x and y coordinates

        reward = self.calculate_reward(elixir, ally_tower_hp, enemy_tower_hp, ally_troop_stats, enemy_troop_stats)

        # Convert all components into a single 1D numpy array
        cards_in_hand_stats = np.array(cards_in_hand_stats)
        elixir = [elixir]
        ally_troop_stats = np.array(ally_troop_stats)
        enemy_troop_stats = np.array(enemy_troop_stats)
        time_remaining = [time_remaining]


        observation = np.concatenate([
            cards_in_hand_stats.flatten(),  # (4 * 13)
            elixir,  # (1)
            ally_troop_stats.flatten(),  # (10 * 13)
            enemy_troop_stats.flatten(),  # (10 * 13)
            ally_tower_hp,  # (3)
            enemy_tower_hp,  # (3)
            time_remaining  # (1)
        ], dtype=np.float32)

        return observation, reward


    def reset(self, seed=None):
        """ Reset the game state """
        super().reset(seed=seed)
        self.screen.leave_game()
        # Restart training battle
        self.screen.start_training_battle()

        start_time = time.time()
        # Expected game end times
        self.end_time = start_time + self.game_time
        self.overtime_end_time = self.end_time + self.overtime_time

        self.previous_ally_towers_destroyed = []
        self.previous_enemy_towers_destroyed = []
        self.previous_min_ally_tower_hp = []
        self.previous_min_enemy_tower_hp = []

        # Sleep until the cards are actually shown on screen
        time.sleep(8)

        obs, reward = self.get_observation()

        return obs, {}


    def step(self, action):
        """ Execute one step in the environment """
        card_index, x, y = action

        self.ai.make_move(card_index, x, y)

        time.sleep(2)

        terminated = self.screen.game_over_check()
        obs, reward = self.get_observation()

        if terminated:
            time.sleep(8)
            winner = self.screen.game_winner_check()
            if winner:
                win_reward = 1000
            else:
                win_reward = -1000
        else:
            win_reward = 0
            
        return obs, reward+win_reward, terminated, False, {}

    def calculate_reward(self, elixir, ally_tower_hp, enemy_tower_hp, ally_troop_stats, enemy_troop_stats):
        """ Define a reward function based on game state """

        # Find the lowest hp ally tower and how many have been destroyed
        ally_towers_destroyed = 0
        min_ally_tower_hp = 4000
        for hp in ally_tower_hp:
            if hp == 0:
                ally_towers_destroyed += 1
            else:
                min_ally_tower_hp = min(min_ally_tower_hp, hp)

        # Find the lowest hp enemy tower and how many have been destroyed
        enemy_towers_destroyed = 0
        min_enemy_tower_hp = 4000
        for hp in enemy_tower_hp:
            if hp == 0:
                enemy_towers_destroyed += 1
            else:
                min_enemy_tower_hp = min(min_enemy_tower_hp, hp)

        self.previous_min_ally_tower_hp.append(min_ally_tower_hp)
        self.previous_min_enemy_tower_hp.append(min_enemy_tower_hp)
        self.previous_ally_towers_destroyed.append(ally_towers_destroyed)
        self.previous_enemy_towers_destroyed.append(enemy_towers_destroyed)

        for list in [self.previous_min_ally_tower_hp, self.previous_min_enemy_tower_hp, self.previous_ally_towers_destroyed, self.previous_enemy_towers_destroyed]:
            if len(list) > 10:
                list.pop(0)

        if len(self.previous_min_ally_tower_hp) == 10: 
            delta_ally_tower_hp = self.previous_min_ally_tower_hp[-1] - self.previous_min_ally_tower_hp[0]
            delta_enemy_tower_hp = self.previous_min_enemy_tower_hp[-1] - self.previous_min_enemy_tower_hp[0]
            delta_ally_towers_destroyed = self.previous_ally_towers_destroyed[-1] - self.previous_ally_towers_destroyed[0]
            delta_enemy_towers_destroyed = self.previous_enemy_towers_destroyed[-1] - self.previous_enemy_towers_destroyed[0]

            # Sometimes towers may mistakenly be marked as lost and will come back again, this change back should be ignored
            if delta_enemy_towers_destroyed < 0:
                delta_enemy_towers_destroyed = 0
            if delta_ally_towers_destroyed < 0:
                delta_ally_towers_destroyed = 0

            tower_difference_reward = 1400 * (delta_enemy_towers_destroyed - delta_ally_towers_destroyed)
            min_tower_difference_reward = delta_ally_tower_hp - delta_enemy_tower_hp
        
        else:
            tower_difference_reward = 0
            min_tower_difference_reward = 0


        troop_hp_difference = 0
        # Remove the troop stats that are padded with 0's
        # Also add the hp difference between troops on the screen
        for i in range(len(ally_troop_stats)):
            if ally_troop_stats[i] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                ally_troop_stats = ally_troop_stats[:i]
                break
            else:
                troop_hp_difference += ally_troop_stats[i][0]

        for i in range(len(enemy_troop_stats)):
            if enemy_troop_stats[i] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                enemy_troop_stats = enemy_troop_stats[:i]
                break
            else:
                troop_hp_difference -= enemy_troop_stats[i][0]

        troop_hp_difference = troop_hp_difference / 20

        # # Reward for the difference in towers destroyed
        # tower_difference_reward = 1400 * (enemy_towers_destroyed - ally_towers_destroyed)

        # # Reward for the difference in minimum tower hp
        # min_tower_difference_reward = min_ally_tower_hp - min_enemy_tower_hp

        # # Reward for the difference in troops on the screen
        # troop_difference_reward = (len(ally_troop_stats) - len(enemy_troop_stats)) * 30

        # Reward for elixir the player has remaining
        elixir_reward = elixir ** 2

        return elixir_reward + troop_hp_difference + tower_difference_reward + min_tower_difference_reward
