import pygetwindow as gw
import time

from Screen import Screen
from Cards import Cards
from AI import AI

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

# Get the current menu screen which can be used to get the current players deck
menu_screen = screen.get_menu_screen()
deck_info = screen.get_deck_info(menu_screen, cards.card_info)

# Start a training battle for testing purposes
screen.start_training_battle()

start_time = time.time()
# A game lasts 180 seconds if it does not go to overtime
game_time = 180
# Overtime lasts 120 seconds if game is not decisive
overtime_time = 120

# Expected game end times
end_time = start_time + game_time
overtime_end_time = end_time + overtime_time

# Initially we are not in overtime and game is not finished
overtime = False
game_over = False

# Sleep until the cards are actually shown on screen
time.sleep(8)

# Getting the players current hand will only be done at the start of the match
# After this tracking of the card cycle can be done manually to prevent the need for more screenshots
cards_in_hand = screen.get_cards_in_hand(deck_info)

# Get the card stats of the players starting hand
cards_in_hand_stats = []
for card in cards_in_hand:
    cards_in_hand_stats.append(cards.get_card_stats(card))

print(cards_in_hand)

while not game_over:
    found_troops = screen.detect_troops()

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
                troop_stats = cards.get_troop_stats(troop_name)
                troop_stats.append(x)
                troop_stats.append(y)
                ally_troop_stats.append(troop_stats)


            elif troop_type == "enemy":
                # Remove enemy_ prefix
                troop_name = troop_name[6:]

                # Create troop_stats array and append to enemy_troop_stats
                troop_stats = cards.get_troop_stats(troop_name)
                troop_stats.append(x)
                troop_stats.append(y)
                enemy_troop_stats.append(troop_stats)


    # test tower hp detection
    ally_tower_hp, enemy_tower_hp = screen.get_tower_hp()
    print(ally_tower_hp)
    print(enemy_tower_hp)

    # test elixir detection
    elixir = screen.get_elixir_count()
    print(elixir)

    if overtime == False:
        # time remaining in game
        time_remaining = end_time - time.time()
        # Check if we are in overtime
        if time_remaining <= 0:
            overtime = True

    else:
        # time remaining in overtime
        time_remaining = overtime_end_time - time.time()
    
    print(time_remaining)

    # Data to pass into AI: ally_troop_stats, ally_tower_hp, enemy_troop_stats, enemy_tower_hp, time_remaining, elixir, cards_in_hand_stats
    # Data output: integer 0-3 indicating card to select in current hand, x and y coordinates

    if elixir == 10:
        ai.make_random_move()
        time.sleep(1)

    game_over = screen.game_over_check()

# Wait until final screen animation finishes
time.sleep(8)

# Check which player won
winner = screen.game_winner_check()
if winner:
    print("Player won the game")

else:
    print("Player lost the game")