import pyautogui
import random

class AI:
    '''
    The agent that interacts with the game
    '''

    def __init__(self):
        # x, y locations of cards the player can play are at from left to right
        self.card_locations = [[840, 1025], [955, 1025], [1070, 1025], [1185, 1025]]

        # borders of the game board
        self.board_left = 705
        self.board_right = 1215
        self.board_bottom = 875
        self.board_middle = 530
        self.board_top = 130


    def make_move(self, card : int, x : int, y : int):
        '''
        Method to make a move that the AI wants to make 

        Parameters:
            card: Which card number in the current hand to play

            x: X coordinate of where to play the card

            y: Y coordinate of where to play the card
        '''

        # AI has chosen to do nothing
        if card == 0:
            return
        
        # Make card point to correct index
        card -= 1
        # Perform the action
        pyautogui.click(self.card_locations[card])
        pyautogui.click(x + self.board_left, y + self.board_top)

    def make_random_move(self):
        '''
        Makes a random move for testing
        '''

        card = random.randint(0, 4)
        x = random.randint(0, self.board_right - self.board_left)
        y = random.randint(0, self.board_bottom - self.board_top)

        self.make_move(card, x, y)
