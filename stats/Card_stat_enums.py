from enum import Enum

# Enumerations used for one-hot-encoding of troop stats in the game
class Card_Speed(Enum):
    NONE = 0
    SLOW = 1
    MEDIUM = 2
    FAST = 3
    VERY_FAST = 4

class Card_Target(Enum):
    GROUND = 0
    AIR_GROUND = 1
    BUILDINGS = 2

class Card_Melee_Range(Enum):
    NONE = 0
    SHORT = 1
    MEDIUM = 2
    LONG = 3

class Card_Range(Enum):
    NONE = 0

class Card_Spell(Enum):
    FALSE = 0
    TRUE = 1
