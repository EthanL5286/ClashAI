import os
import cv2
from cv2.typing import MatLike
import dxcam
import numpy as np
from numpy.typing import NDArray
import pyautogui
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
import easyocr

class Screen():
    '''
    Class that controls everything to do with detecting images on the screen and details about the screen
    '''

    def __init__(self, region: tuple[int, int, int, int]):
        self.region = region
        self.camera = dxcam.create()

        # Image identifiers used to tell what menu screen we are currently looking at
        self.identifiers = self.load_screen_identifiers()
        # Machine learning model for troop detection
        self.model = YOLO("troop_detector.pt")
        # Easyocr text recognition reader
        self.reader = easyocr.Reader(['en'])
        # Kernel used for sharpening images for text recognition
        self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # All menu locations are assuming that the current state is in the shop
        self.shop_location = [20 + region[0], 1080 + region[1]]
        self.collection_location = [260 + region[0], 1080 + region[1]]
        self.battle_location = [370 + region[0], 1080 + region[1]]
        self.menu_bar_location = [575 + region[0], 148 + region[1]]
        self.training_camp_location = [385 + region[0], 397 + region[1]]
        self.training_camp_ok_location = [435 + region[0], 675 + region[1]]

        self.ally_tower_hp_regions = [(1101, 723, 1150, 745), (775, 723, 825, 745), (940, 872, 995, 892)]
        self.enemy_tower_hp_regions = [(1101, 176, 1150, 205), (775, 176, 825, 205), (942, 48, 995, 68)]

        self.elixir_bar_width = 45
        self.elixir_bar_location = [1244, 1120]

    def load_screen_identifiers(self) -> dict[str, MatLike]:
        '''
        Loads the images used to check what screen the player is on

        Returns:
            screen_identifiers: Dictionary where key is the screen name and value is the image of the screen identifier
        '''

        screen_identifiers = {}
        for img_name in os.listdir("./screen_identifiers"):
            img = cv2.imread("screen_identifiers/{}".format(img_name))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_name = img_name.split(".")[0]
            screen_identifiers[img_name] = gray_img

        return screen_identifiers
    
    def take_screenshot(self, region: tuple[int, int, int, int] | None = None) -> NDArray:
        '''
        Takes a screenshot of the current game screen and returns the image

        Parameters:
            region: The region for the screenshot to be taken, if None then self.region is used

        Returns:
            screenshot: Screenshot of the region
        '''
        screenshot = np.array(None)

        #TODO: Maybe consider just discarding a screenshot if it fails for efficiency
        while (screenshot is None) or (screenshot.size == 0) or (screenshot.shape == ()):

            if region is None:
                screenshot = np.array(self.camera.grab(region=(self.region)))
            else:
                screenshot = np.array(self.camera.grab(region=(region)))

        return screenshot
    
    def get_menu_screen(self) -> str:
        '''
        Gets the menu screen that the player is currently looking at

        Returns:
            menu_screen: The name of the current menu on screen  
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
    
    def get_deck_info(self, menu_screen: str, card_info: dict[str, MatLike]) -> dict[str, MatLike]:
        '''
        Gets the deck the player is using by navigating to the deck screen and matching card images against it

        Parameters:
            menu_screen: Name of the current menu screen the player is on
            card_info: Dictionary where key is the card name and value is the card image

        Returns:
            deck_info: Subset of the card info dictionary only containing cards in the current players deck 
        
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
    
    def start_training_battle(self):
        '''
        Clicks a set of buttons to initiate a training camp battle
        '''
        pyautogui.click(self.shop_location)
        time.sleep(1)
        pyautogui.click(self.battle_location)
        time.sleep(1)
        pyautogui.click(self.menu_bar_location)
        time.sleep(1)
        pyautogui.click(self.training_camp_location)
        time.sleep(1)
        pyautogui.click(self.training_camp_ok_location)
    
    def get_cards_in_hand(self, deck_info: dict[str, MatLike]) -> list[str]:
        '''
        Gets the cards in the players hand by matching against images of cards in the players deck
        
        Parameters: 
            deck_info: Dictionary where key is the card name and value is the card image

        Returns:
            cards_in_hand: Array of strings relating to the cards in the players hand from left to right
        
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
    
    def get_tower_hp(self) -> tuple[list[int], list[int]]:
        '''
        Takes screenshots of each crown tower hp bar and uses easyocr to detect the number values

        Returns:
            ally_tower_hp: List containing each ally tower hp value

            enemy_tower_hp: List containing each enemy tower hp value
        '''
        ally_tower_hp = []
        enemy_tower_hp = []

        for tower_region in self.ally_tower_hp_regions:
            screenshot = self.take_screenshot(tower_region)
            # Change to RGB as easyocr seems to work better in RGB
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            # Sharpen the image to pronounce the edges of the numbers
            screenshot = cv2.filter2D(screenshot, -1, self.kernel)

            # Extract the hp if digits are found
            text = self.reader.readtext(screenshot)
            if text:
                hp = str(text[0][1])
                if hp.isdigit():
                    ally_tower_hp.append(int(hp))

        for tower_region in self.enemy_tower_hp_regions:
            screenshot = self.take_screenshot(tower_region)
            # Change to RGB as easyocr seems to work better in RGB
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            # Sharpen the image to pronounce the edges of the numbers
            screenshot = cv2.filter2D(screenshot, -1, self.kernel)

            # Extract the hp if digits are found
            text = self.reader.readtext(screenshot)
            if text:
                hp = str(text[0][1])
                if hp.isdigit():
                    enemy_tower_hp.append(int(hp))

        return ally_tower_hp, enemy_tower_hp
    
    def get_elixir_count(self) -> int:
        '''
        Gets the current elixir count of the player by checking pixel colours in the elixir bar

        Returns:
            elixir_count: Integer amount of elixir the player has
        '''
        # Check pixels from right hand side of elixir bar so start at 10 elixir
        elixir_count = 10

        # Move back an elixir_bar_width each time we can't detect the players elixir
        for i in range(0, 10):
            x = self.elixir_bar_location[0] - (i * self.elixir_bar_width)
            y = self.elixir_bar_location[1]

            #TODO: Could maybe take a single screenshot and get multiple pixel values at once to improve efficiency
            # This pixel colour is the background of the elixir bar so if false then we return the elixir amount
            if pyautogui.pixelMatchesColor(x, y, (5,53,122), tolerance=50):
                elixir_count -= 1
            else:
                return elixir_count
            
        return 0

    def detect_troops(self) -> list[Results]:
        '''
        Takes a screenshot of the current game screen and uses the machine learning model to predict where troops are on the screen

        Returns:
            results: List containing the information for each detected troop
        '''
        screenshot = self.take_screenshot()
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=screenshot, stream=True, conf=0.4, verbose=False)

        return results

        # These comments are left in for future reference about where data is held in the result
        # for result in results:
        #     result.show()
            # print(result.boxes.xywh)
            # print(result.boxes.cls)
            # print(result.boxes.cls[0].item())
            # print(result.boxes.xywh[0].numpy())

            # print("there is a : {} at position: x = {}, y = {}".format(result.names[result.boxes.cls[0].item()], result.boxes.xywh[0][0], result.boxes.xywh[0][1]))
