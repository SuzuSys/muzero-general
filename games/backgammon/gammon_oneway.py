# Following function always moves characters 
# from high index to low index.

import random, enum
from typing import NamedTuple, Optional, Tuple, List
import dataclasses
import numpy as np

import collections

from .config import PIPS_COUNT, CHECKERS, HOME_POINTS, POINTS
from .position import Position
PositionType = Position
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

class GammonOneway:
    def __init__(self, seed: int):
        """
        Initialise Backgammon.

        Args:
            seed (int): random seed
        """
        random.seed(seed)
        self.position: PositionType = Position()
        self.legal_plays: List[MoveType] = []
        self.dice: DiceType = Dice(0,0)
        self.used_dice: UsedDiceType = UsedDice()
    
    def swap_players(self) -> None:
        """
        Invert position. and reset used_dice
        """
        self.position = self.position.swap_players()
        self.used_dice = UsedDice()
    
    def reset(self) -> None:
        """
        Reset the game.
        """
        self.position = Position()
        self.legal_plays = []
        self.dice = Dice(0,0)
        self.used_dice = UsedDice()
    
    def roll(self) -> None:
        """
        Dice roll.
        """
        self.dice = Dice(
            random.randrange(1, PIPS_COUNT),
            random.randrange(1, PIPS_COUNT),
        )
    
    def first_roll(self) -> None:
        """
        Dice roll at the beginning of the game.\n
        This function avoids rolling doubles.
        """
        while True:
            self.roll()
            if self.dice.left != self.dice.right:
                return
    
    def generate_plays(self) -> None:
        """
        Generate all combinations of legal plays.
        """

        def generate(
            position: PositionType,
            dice: Tuple[int, ...],
            die: int,
        ) -> Tuple[List[MoveType], int]:
            """
            Get all combinations of legal plays recursively when using a left die first.

            Args:
                position (PositionType): current position
                dice (Tuple[int, ...]): dice
                die (int): current dice's index

            Returns:
                plays (List[MoveType]): legal moves
                depth (int): length of series of moves
            """

            if die == len(dice):
                return ([], 0)

            new_position: Optional[PositionType]
            destination: Optional[int]
            point: int
            move: MoveType
            child_plays: List[MoveType]
            depth: int
            max_depth: int = 0 # max depth of child nodes
            plays: List[MoveType] = []

            pips: int = dice[die]

            if position.player_bar > 0:
                # enter
                new_position, destination = position.enter(pips)
                if new_position:
                    child_plays, max_depth = generate(
                        new_position,
                        dice,
                        die + 1,
                    )
                    move = Move(pips, None, destination, child_plays)
                    plays.append(move)
            elif sum(position.player_home()) + position.player_off == CHECKERS:
                # bearing off
                for point in range(HOME_POINTS):
                    new_position, destination = position.off(point, pips)
                    if new_position:
                        child_plays, depth = generate(
                            new_position,
                            dice,
                            die + 1
                        )
                        if max_depth <= depth:
                            move = Move(pips, point, destination, child_plays)
                            if max_depth < depth:
                                plays = [move]
                                max_depth = depth
                            else:
                                plays.append(move)
            else:
                # normal play
                for point in range(POINTS):
                    new_position, destination = position.move(point, pips)
                    if new_position:
                        child_plays, depth = generate(
                            new_position,
                            dice,
                            die + 1
                        )
                        if max_depth <= depth:
                            move = Move(pips, point, destination, child_plays)
                            if max_depth < depth:
                                plays = [move]
                                max_depth = depth
                            else:
                                plays.append(move)
            if len(plays) > 0:
                return plays, max_depth + 1
            return [], 0
        
        doubles: bool = self.dice.left == self.dice.right
        dice: Tuple[int, ...] = (self.dice.left,) * 4 if doubles \
            else (self.dice.left, self.dice.right)
        plays: List[MoveType]
        if doubles:
            plays, _ = generate(self.position, dice, 0)
        else:
            left_plays: List[MoveType]
            left_depth: int
            left_plays, left_depth = generate(self.position, dice, 0)
            right_plays: List[MoveType]
            right_depth: int
            right_plays, right_depth = generate(self.position, dice[::-1], 0)
            if left_depth < right_depth:
                plays = right_plays
            elif right_depth < left_depth:
                plays = left_plays
            elif left_depth == 1:
                if self.dice.left < self.dice.right:
                    plays = right_plays
                else:
                    plays = left_plays
            else:
                plays = left_plays + right_plays
        self.legal_plays = plays

    def get_legal_actions(self) -> List[int]:
        """
        Return an array of integers, subset of the action space.

        Returns:
            List[int]: an array of integers, subset of the action space 
        """
        if len(self.legal_plays) == 0:
            return [0]

        def pips_to_die(dice: DiceType, pips: int) -> DieType:
            if pips == dice.left:
                return Die.LEFT
            elif pips == dice.right:
                return Die.RIGHT

        legal_moves: List[ActionType] = [
            Action(pips_to_die(self.dice, move.pips), move.source)
            for move in self.legal_plays]
        
        def source_to_action(die: DieType, source: Optional[int]) -> int:
            if source:
                return (POINTS + 1) * die + source + 1
            else:
                return (POINTS + 1) * (die + 1)
        
        legal_actions: List[int] = [source_to_action(action.die, action.source)
            for action in legal_moves]

        return legal_actions
    
    def action(self, action: int) -> MatchStateType:
        """
        Apply action.

        Args:
            action (int): index of the action space

        Returns:
            MatchState: match state
        """
        die: DieType
        source: int
        if action == 0:
            return MatchState.NOTHING
        elif action <= POINTS + 1:
            die = Die.LEFT
            action -= 1
        else:
            die = Die.RIGHT
            action -= POINTS + 2
        if action == POINTS:
            source = None
        else:
            source = action
        
        pips: int = self.dice[die]
        move: Move
        for m in self.legal_plays:
            if m.pips == pips and m.source == source:
                move = m
                break
        self.position = self.position.apply_move(move.source, move.destination)
        self.legal_plays = move.next_moves
        self.used_dice.count(die)

        match_state: MatchStateType = MatchState.NOTHING
        if len(self.legal_plays) == 0:
            if self.position.player_off == CHECKERS:
                match_state = MatchState.GAME_OVER
            else:
                match_state = MatchState.TURN_OVER

        return match_state

        

        
        
        
        



                    

