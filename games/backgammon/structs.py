import enum
from typing import NamedTuple, Optional, List
import dataclasses
import numpy as np

NumpyType = np.ndarray

@enum.unique
class Die(enum.IntEnum):
    LEFT = 0b00
    RIGHT = 0b01
DieType = Die

class Dice(NamedTuple):
    left: int
    right: int
DiceType = Dice

@dataclasses.dataclass
class UsedDice:
    left: int = 0
    right: int = 0
    def count(self, die: DieType) -> None:
        if die == Die.LEFT:
            self.left += 1
        else:
            self.right += 1
UsedDiceType = UsedDice

@enum.unique
class MatchState(enum.Enum):
    NOTHING = 0b00
    TURN_OVER = 0b01
    GAME_OVER = 0b10
MatchStateType = MatchState

@dataclasses.dataclass
class Move:
    pips: int
    source: Optional[int]
    destination: Optional[int]
    next_moves: List["Move"]
MoveType = Move

class Action(NamedTuple):
    die: DieType
    source: Optional[int]
ActionType = Action

@enum.unique
class Player(enum.IntEnum):
    ZERO = 0b00
    ONE = 0b01
PlayerType = Player