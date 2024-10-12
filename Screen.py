import os
import cv2
import dxcam
import numpy as np
import pyautogui
import time

class Screen():
    '''
    Class that controls everything to do with detecting images on the screen and details about the screen
    '''

    def __init__(self, region):
        self.region = region
        self.camera = dxcam.create()
        self.identifiers = self.load_screen_identifiers()

        # All menu locations are assuming that the current state is in the shop
        self.shop_location = [20 + region[0], 1080 + region[1]]
        self.collection_location = [260 + region[0], 1080 + region[1]]

    def load_screen_identifiers(self):
        '''
        Loads the images used to check what screen the player is on

        Returns:
            screen_identifiers (dict):
                Dictionary where key is the screen name and value is the image of the screen identifier
        '''

        screen_identifiers = {}
        for img_name in os.listdir("./screen_identifiers"):
            img = cv2.imread("screen_identifiers/{}".format(img_name))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_name = img_name.split(".")[0]
            screen_identifiers[img_name] = gray_img

        return screen_identifiers
    
    def take_screenshot(self):
        '''
        Takes a screenshot of the current game screen and returns the image

        Returns:
            screenshot (np array):
                Screenshot of the current game screen region determined by self.region
        '''

        return np.array(self.camera.grab(region=(self.region)))
    
    def get_menu_screen(self):
        '''
        Gets the menu screen that the player is currently looking at

        Returns:
            menu_screen (str):
                The name of the current menu on screen  
        '''

        menu_screen = "undefined"

        screenshot = self.take_screenshot()
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        for key in self.identifiers:
            result = cv2.matchTemplate(gray_screenshot, self.identifiers[key], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= 0.8:
                # Detected a screen identifier
                menu_screen = key
                break

        return menu_screen
    
    def get_deck_info(self, menu_screen, card_info):
        '''
        Gets the deck the player is using by navigating to the deck screen and matching card images against it

        Parameters:
            menu_screen (str):
                Name of the current menu screen the player is on
            card_info (dict):
                Dictionary where key is the card name and value is the card image

        Returns:
            deck_info (dict):
                Subset of the card images dictionary only containing cards in the current players deck 
        
        '''
        deck_info = {}

        # The amount of resizing needed to detect cards on the deck screen
        card_image_resize = 1.35

        if menu_screen != "collection_screen" and menu_screen != "undefined":
            # On a menu screen that isn't the collection screen
            # Menu bar moves depending on current page so calibrate by going to shop
            pyautogui.click(self.shop_location)
            time.sleep(1)
            pyautogui.click(self.collection_location)
            time.sleep(1)
            menu_screen = "collection_screen"

        if menu_screen == "collection_screen":
            # Currently looking at the deck, do image detection
            cards_found = 0
            screenshot = self.take_screenshot()
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            for card_name in card_info:
                # Resize the template according to the scale
                resized_card = cv2.resize(card_info[card_name], (0, 0), fx=card_image_resize, fy=card_image_resize)
                result = cv2.matchTemplate(gray_screenshot, resized_card, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val >= 0.8:
                    # Detected a card
                    deck_info[card_name] = card_info[card_name]
                    cards_found +=1
                    print(cards_found)
                    if cards_found >= 8:
                        break

        return deck_info
    
    def get_cards_in_hand(self, deck_info):
        '''
        Gets the cards in the players hand by matching against images of cards in the players deck
        
        Parameters: 
            deck_info (dict): 
                Dictionary where key is the card name and value is the card image

        Returns:
            cards_in_hand (array):
                Array of strings relating to the cards in the players hand from left to right
        
        '''
        cards_in_hand = []
        cards_found = 0

        # The amount of resizing needed to detect cards on the battle screen
        card_image_resize = 1.15

        screenshot = self.take_screenshot()
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        for card_name in deck_info:
            # Resize the template according to the scale
            resized_card = cv2.resize(deck_info[card_name], (0, 0), fx=card_image_resize, fy=card_image_resize)
            result = cv2.matchTemplate(gray_screenshot, resized_card, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= 0.8:
                # Detected a card
                cards_in_hand.append([max_loc, card_name])
                cards_found +=1
                if cards_found >= 4:
                    break
                
        # Sort the cards based on their x location to get cards in order of left to right
        sorted_cards_in_hand = sorted(cards_in_hand, key=lambda x: x[0][0])
        cards_in_hand = [name for location, name in sorted_cards_in_hand]

        return cards_in_hand