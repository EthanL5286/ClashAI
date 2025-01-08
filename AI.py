import pyautogui
import random

class AI:
    '''
    The AI for the clash bot
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
        pyautogui.click(self.card_locations[card])
        pyautogui.click(x, y)

    def make_random_move(self):
        '''
        Makes a random move for testing
        '''

        card = random.randint(0, 3)
        x = random.randint(self.board_left, self.board_right)
        y = random.randint(self.board_middle, self.board_bottom)

        self.make_move(card, x, y)
