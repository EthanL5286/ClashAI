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

        # Get the card order for the AI to pick a specific card from index 0-8
        self.card_order = []
        for card_name in self.deck_info.keys():
            self.card_order.append(card_name)

        # A game lasts 180 seconds if it does not go to overtime
        self.game_time = 180
        # Overtime lasts 120 seconds if game is not decisive
        self.overtime_time = 120

        # Initially end times are 0
        self.end_time = 0
        self.overtime_end_time = 0

        # Initially we are not in overtime and game is not finished
        self.overtime = False

        # Check that the previous move was actually valid
        self.invalid_move = False
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=6000, shape=(
                (4 * 13) +              # 4 cards in hand, each with 13 stats
                1 +                     # elixir
                (max_troops * 13) +     # max ally troops, each with 13 stats
                (max_troops * 13) +     # max enemy troops, each with 13 stats
                max_towers +            # max number of ally towers
                max_towers +            # max number of enemy towers
                1                       # time remaining
            ,), dtype=np.float32
        )

        # Action space is the choice of 5 options (nothing, card1, card2, card3, card4) with the x and y coordinates
        self.action_space = gym.spaces.MultiDiscrete([9, 20, 30])

        # Reward based on change in tower hp over time
        self.previous_min_ally_tower_hp = []
        self.previous_min_enemy_tower_hp = []
        self.previous_ally_towers_destroyed = []
        self.previous_enemy_towers_destroyed = []

    def get_observation(self) -> tuple[list[float], int]:
        '''
        Gets the observations from the screen

        Returns:
            observation: a list of game data flattened into a single array
            reward: An integer value determining the reward of the current state
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

        # Pad results with 0's or remove data if too large
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

        # Get the tower hp for both players
        ally_tower_hp, enemy_tower_hp = self.screen.get_tower_hp()

        # Get the players elixir count
        elixir = self.screen.get_elixir_count()

        # Get the time remaining in game
        if self.overtime == False:
            time_remaining = self.end_time - time.time()
            # Check if we are in overtime
            if time_remaining <= 0:
                self.overtime = True
        else:
            time_remaining = self.overtime_end_time - time.time()

        # Send collected data to calculate reward for current state
        reward = self.calculate_reward(elixir, ally_tower_hp, enemy_tower_hp, ally_troop_stats, enemy_troop_stats)

        # Convert all components into 1D numpy arrays
        cards_in_hand_stats = np.array(cards_in_hand_stats)
        elixir = [elixir]
        ally_troop_stats = np.array(ally_troop_stats)
        enemy_troop_stats = np.array(enemy_troop_stats)
        time_remaining = [time_remaining]

        # Combine all 1D arrays into a single 1D array
        observation = np.concatenate([
            cards_in_hand_stats.flatten(),  # 4 cards in hand, each with 13 stats
            elixir,                         # elixir
            ally_troop_stats.flatten(),     # max ally troops, each with 13 stats
            enemy_troop_stats.flatten(),    # max enemy troops, each with 13 stats
            ally_tower_hp,                  # max number of ally towers
            enemy_tower_hp,                 # max number of enemy towers
            time_remaining                  # time remaining
        ], dtype=np.float32)

        return observation, reward


    def reset(self, seed=None):
        """
        Resets the game state when a game finishes to start another battle against the AI

        Parameters: 
            seed: Seed is by default None and is only used to call the super function

        Returns:
            observation: The observation of the first game state after the game is reset

            info: An information dictionary which is not used so a blank dictionary is returned
        """
        super().reset(seed=seed)
        # Leave the final game screen to get back to the main menu
        self.screen.leave_game()
        # Restart training battle
        self.screen.start_training_battle()
        
        # Restart the game timer
        start_time = time.time()
        self.end_time = start_time + self.game_time
        self.overtime_end_time = self.end_time + self.overtime_time
        self.overtime = False

        # Reset arrays used for calculating reward
        self.previous_ally_towers_destroyed = []
        self.previous_enemy_towers_destroyed = []
        self.previous_min_ally_tower_hp = []
        self.previous_min_enemy_tower_hp = []

        # Sleep until the cards are actually shown on screen
        time.sleep(8)

        # Get the initial observation of the game state
        observation, _ = self.get_observation()

        return observation, {}


    def step(self, action: tuple[int, int, int]) -> tuple[list[float], int, bool, bool, dict]:
        """ 
        Execute one step in the environment which consists of playing a card at a position or choosing not to move

        Parameters:
            action: The action chosen by the AI consisting if a card index indicating which card to play and (x, y) of where to play it

        Returns:
            observation: The state after the move has been played
            reward: The reward after the given move
            terminated: Boolean determining if the game has finished after this step
            truncated: Always set to False as game cannot be truncated early
            info: An information dictionary which is not used so a blank dictionary is returned
        
        """
        # Extract values from action
        card_index, x, y = action

        # Get the name of the card chosen
        if card_index == 0:
            card_chosen = None
        else:
            card_chosen = self.card_order[card_index - 1]

        if card_chosen is not None:
            # Check if the move is valid by being in the hand and having enough elixir to play it
            cards_in_hand = self.screen.get_cards_in_hand(self.deck_info)
            if card_chosen in cards_in_hand:
                elixir_cost = self.cards.get_card_stats(card_chosen)[-1]
                if elixir_cost <= self.screen.get_elixir_count():
                    # Reset the card index as the move is valid
                    card_index = cards_in_hand.index(card_chosen) + 1
                else:
                    self.invalid_move = True
            else:
                self.invalid_move = True

        # Map coordinates grid to x,y values
        x = x * 25
        y = y * 25

        # Make the move if it is not invalid
        if not self.invalid_move:
            self.ai.make_move(card_index, x, y)

        # Wait to let the game state settle to provide better observations
        time.sleep(2)

        # Check if the game is over and get the observation and reward for the move
        terminated = self.screen.game_over_check()
        observation, reward = self.get_observation()

        # If the game has finished wait for the winner screen to show to determine winner
        if terminated:
            time.sleep(8)
            winner = self.screen.game_winner_check()
            if winner:
                win_reward = 5000
            else:
                win_reward = -5000
        else:
            win_reward = 0
            
        return observation, reward+win_reward, terminated, False, {}

    def calculate_reward(self, elixir: int, ally_tower_hp: list[int], enemy_tower_hp: list[int], 
                         ally_troop_stats: list[float], enemy_troop_stats: list[float]) -> float:
        """ 
        Calculates a reward based on information provided by observation

        Parameters: 
            elixir: The players elixir count
            ally_tower_hp: List of all 3 ally towers and their hp values
            enemy_tower_hp: List of all 3 enemy towers and their hp values
            ally_troop_stats: List of the stats of the ally troops that are currently on the screen
            enemy_troop_stats: List of the stats of the enemy troops that are currently on screen

        Returns: 
            reward: Float value for the reward of the current state
        """

        # Punish for not picking a valid move
        if self.invalid_move:
            self.invalid_move = False
            return -5000

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

        # Add the minimum tower hp and towers destroyed to the list of previous values
        # Previous values are stored so that the reward function can be spread for a longer timeframe reward based on change in hp over time
        # This helps promote long term goals rather than immediate reward that could lead to losing the game
        self.previous_min_ally_tower_hp.append(min_ally_tower_hp)
        self.previous_min_enemy_tower_hp.append(min_enemy_tower_hp)
        self.previous_ally_towers_destroyed.append(ally_towers_destroyed)
        self.previous_enemy_towers_destroyed.append(enemy_towers_destroyed)

        # Remove the first value like a FIFO queue so the size is always 10
        for list in [self.previous_min_ally_tower_hp, self.previous_min_enemy_tower_hp, self.previous_ally_towers_destroyed, self.previous_enemy_towers_destroyed]:
            if len(list) > 10:
                list.pop(0)

        # If we have 10 values then start calculating the reward based on change in tower hp from the first to last position
        if len(self.previous_min_ally_tower_hp) == 10: 
            # Change in hp
            delta_ally_tower_hp = self.previous_min_ally_tower_hp[-1] - self.previous_min_ally_tower_hp[0]
            delta_enemy_tower_hp = self.previous_min_enemy_tower_hp[-1] - self.previous_min_enemy_tower_hp[0]

            # Change in towers destroyed
            delta_ally_towers_destroyed = self.previous_ally_towers_destroyed[-1] - self.previous_ally_towers_destroyed[0]
            delta_enemy_towers_destroyed = self.previous_enemy_towers_destroyed[-1] - self.previous_enemy_towers_destroyed[0]

            # Sometimes towers may mistakenly be marked as lost and will come back again, this change back should be ignored
            if delta_enemy_towers_destroyed < 0:
                delta_enemy_towers_destroyed = 0
            if delta_ally_towers_destroyed < 0:
                delta_ally_towers_destroyed = 0

            # Reward is simply the amount of hp difference change between ally towers and enemy towers
            tower_difference_reward = 1400 * (delta_enemy_towers_destroyed - delta_ally_towers_destroyed)
            min_tower_difference_reward = delta_ally_tower_hp - delta_enemy_tower_hp
        
        # If we dont have 10 values yet then don't give a reward
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

        # Troop hp difference is weighted less as tower hp is more important but helps to guide the AI to kill enemy troops
        troop_hp_difference = troop_hp_difference / 20

        # Reward for elixir the player has remaining
        elixir_reward = elixir ** 2

        return elixir_reward + troop_hp_difference + tower_difference_reward + min_tower_difference_reward
