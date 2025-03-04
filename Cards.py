import os
import cv2
from cv2.typing import MatLike
import json
class Cards:
    '''
    Class that takes care of all information regarding the cards
    '''

    def __init__(self):
        # card_info stores the images and name pairs for each card
        self.card_info = self.load_card_info()

        # card and troop stats hold data for each card or troop based on in game stats
        self.card_stats = json.load(open("stats/card_stats.json", "r"))
        self.troop_stats = json.load(open("stats/troop_stats.json", "r"))

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
    
    def get_card_stats(self, card_name: str) -> list[float]:
        '''
        Gets the cards stats from the card_stats.json such as hp and elixir

        Parameters:
            card_name: String of the card name to get the stats of

        Returns:
            card_stats: A list of float values for each stat of the card
        
        '''
        stats = self.card_stats[card_name]

        return [float(stats["hp"]), float(stats["direct_damage"]), float(stats["splash_damage"]), float(stats["hit_speed"]), 
                float(stats["speed"]), float(stats["melee_range"]), float(stats["range"]), float(stats["targets"]), float(stats["flying"]), 
                float(stats["spell"]), float(stats["spell_radius"]), float(stats["troop_count"]), float(stats["elixir"])]
    
    def get_troop_stats(self, troop_name: str) -> list[float] | None:
        '''
        Gets the troop stats from the troop_stats.json such as hp and direct damage

        Parameters:
            troop_name: String of the troop name to get the stats of

        Returns:
            troop_stats: A list of float values for each stat of the troop
        
        '''
        if troop_name in self.troop_stats:
            stats = self.troop_stats[troop_name]
        else:
            return None

        return [float(stats["hp"]), float(stats["direct_damage"]), float(stats["splash_damage"]), float(stats["hit_speed"]), 
                float(stats["speed"]), float(stats["melee_range"]), float(stats["range"]), float(stats["targets"]), float(stats["flying"]), 
                float(stats["spell"]), float(stats["spell_radius"])]

            
        