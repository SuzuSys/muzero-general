# Following function always moves characters 
# from high index to low index.

import random
from typing import Optional, Tuple, List
import numpy as np

from .config import PIPS_COUNT, CHECKERS, HOME_POINTS, POINTS
from .position import Position, PositionType
from .structs import Die, Dice, UsedDice, MatchState, Move, DieType, DiceType, UsedDiceType, MatchStateType, MoveType, NumpyType

class GammonOneway:
    def __init__(self, seed: int):
        """
        Initialise Backgammon.\n
        In this class, the checkers are always moved from right to left.

        Args:
            seed (int): random seed
        """
        random.seed(seed)
        self.position: PositionType = Position()
        self.legal_plays: List[MoveType] = []
        self.legal_plays_depth: int = 0
        self.move_number: int = 0
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
            random.randrange(PIPS_COUNT) + 1,
            random.randrange(PIPS_COUNT) + 1,
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
            plays, self.legal_plays_depth = generate(self.position, dice, 0)
        else:
            left_plays: List[MoveType]
            left_depth: int
            left_plays, left_depth = generate(self.position, dice, 0)
            right_plays: List[MoveType]
            right_depth: int
            right_plays, right_depth = generate(self.position, dice[::-1], 0)
            if left_depth < right_depth:
                plays = right_plays
                self.legal_plays_depth = right_depth
            elif right_depth < left_depth:
                plays = left_plays
                self.legal_plays_depth = left_depth
            elif left_depth == 1:
                self.legal_plays_depth = 1
                if self.dice.left < self.dice.right:
                    plays = right_plays
                else:
                    plays = left_plays
            else:
                plays = left_plays + right_plays
                self.legal_plays_depth = left_depth
        self.legal_plays = plays
    
    def get_legal_points(self) -> NumpyType:
        """
        legal_actions to numpy: [g|...|b]
        
        Returns:
            NumpyType: [g|...|b]
            0: Left dice,  source
            1: Left dice,  destination
            2: Right dice, source
            3: Right dice, destination
        """

        def pips_to_die(dice: DiceType, pips: int) -> DieType:
            if pips == dice.left:
                return Die.LEFT
            elif pips == dice.right:
                return Die.RIGHT

        points_array: NumpyType = np.zeros((4, POINTS + 2), dtype='int8')
        for move in self.legal_plays:
            die = pips_to_die(self.dice, move.pips)
            if move.source == None:
                points_array[die*2, POINTS+1] = 1
            else:
                points_array[die*2, move.source+1] = 1
            if move.destination == None:
                points_array[die*2+1, 0] = 1
            else:
                points_array[die*2+1, move.destination+1] = 1
        return points_array
            
    
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
        action_temp: int = action
        if action_temp == 0:
            self.used_dice.count(0)
        else:
            if action_temp <= POINTS + 1:
                die = Die.LEFT
                action_temp -= 1
            else:
                die = Die.RIGHT
                action_temp -= POINTS + 2
            if action_temp == POINTS:
                source = None
            else:
                source = action_temp
            
            pips: int = self.dice[die]
            move: Optional[MoveType] = None
            drive: List[Tuple[int, int]] = []
            for m in self.legal_plays:
                drive.append((m.pips, m.source))
                if m.pips == pips and m.source == source:
                    move = m
                    break
            self.position = self.position.apply_move(move.source, move.destination)
            self.legal_plays = move.next_moves
            self.used_dice.count(pips)

        match_state: MatchStateType = MatchState.NOTHING
        if len(self.legal_plays) == 0:
            if self.position.player_off == CHECKERS:
                match_state = MatchState.GAME_OVER
            elif self.move_number == 3:
                match_state = MatchState.TURN_OVER
                self.move_number = 0
            else:
                self.move_number += 1
        else:
            self.move_number += 1

        return match_state

GammonOnewayType = GammonOneway