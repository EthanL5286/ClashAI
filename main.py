import pygetwindow as gw

from Screen import Screen
from Cards import Cards

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

# Get the current menu screen which can be used to get the current players deck
menu_screen = screen.get_menu_screen()
deck_info = screen.get_deck_info(menu_screen, cards.card_info)

# Start a training battle for testing purposes
screen.start_training_battle()

ready = input("press enter when ready")

# Getting the players current hand will only be done at the start of the match
# After this tracking of the card cycle can be done manually to prevent the need for more screenshots
cards_in_hand = screen.get_cards_in_hand(deck_info)
print(cards_in_hand)

while True:
    found_troops = screen.detect_troops()

    ally_troops = []
    enemy_troops = []

    for result in found_troops:
        for i in range(0, len(result.boxes.cls)):
            x = round(result.boxes.xywh[i][0].item()) + round(result.boxes.xywh[i][2].item() / 2)
            y = round(result.boxes.xywh[i][1].item()) + round(result.boxes.xywh[i][3].item() / 2)
            troop_name = str(result.names[result.boxes.cls[i].item()])

            troop_type = troop_name.split("_")[0]
            if troop_type == "ally":
                ally_troops.append([troop_name, x, y])

            elif troop_type == "enemy":
                enemy_troops.append([troop_name, x, y])

        print(ally_troops)
        print(enemy_troops)


    # test tower hp detection
    ally_tower_hp, enemy_tower_hp = screen.get_tower_hp()
    print(ally_tower_hp)
    print(enemy_tower_hp)

    # test elixir detection
    print(screen.get_elixir_count())