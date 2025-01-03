import os
import cv2
from cv2.typing import MatLike

class Cards:
    '''
    Class that takes care of all information regarding the cards
    '''

    def __init__(self):
        self.card_info = self.load_card_info()

    def load_card_info(self) -> dict[str, MatLike]:
        '''
        Loads the card names and their images ready for matching

        Returns:
            card_info: Dictionary where key is the card name and value is the grayscale image of the card
        '''

        card_info = {}
        for card_name in os.listdir("./card_images"):
            card_img = cv2.imread("card_images/{}".format(card_name))
            gray_card_image = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            card_name = card_name.split(".")[0]
            card_info[card_name] = gray_card_image

        return card_info

            
        