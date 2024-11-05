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

menu_screen = screen.get_menu_screen()
deck_info = screen.get_deck_info(menu_screen, cards.card_info)

ready = input("press enter when ready")

cards_in_hand = screen.get_cards_in_hand(deck_info)

print(cards_in_hand)



