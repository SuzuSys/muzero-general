import numpy as np
import dataclasses
from typing import Optional, Tuple

from .config import INITIAL_POSITION, POINTS, HOME_POINTS
from .structs import NumpyType

@dataclasses.dataclass(frozen=True)
class Position:
    board_points: NumpyType = INITIAL_POSITION
    player_bar: int = 0
    player_off: int = 0
    opponent_bar: int = 0
    opponent_off: int = 0

    def enter(
        self, 
        pips: int
    ) -> Tuple[Optional["Position"], Optional[int]]:
        """
        Try to enter from the bar.

        Args:
            pips (int): pips

        Returns:
            Optional[new Position]: Result of applying the move. If not applicable, return None.
            destination (Optional[int]): A point of the checker after moving. If not applicable, return None.
        """
        destination: int = POINTS - pips
        if self.board_points[destination] >= -1:
            return self.apply_move(None, destination), destination
        return None, None
    
    def player_home(self) -> np.ndarray:
        """
        Return player's checkers in the player's home board.

        Returns:
            new np.ndarray(size=HOME_BOARD)
        """
        home_board: np.ndarray = self.board_points[:HOME_POINTS].copy()
        home_board[home_board < 0] = 0
        return home_board
    
    def off(
        self, 
        point: int, 
        pips: int
    ) -> Tuple[Optional["Position"], Optional[int]]:
        """
        Try to move a checker in the player's home board

        Args:
            point (int): A point of the checker before moving
            pips (int): pips

        Returns:
            Optional[new Position]: Result of applying the move. If not applicable, return None.
            destination (Optional[int]): A point of the checker after moving. None meaning player off.
        """
        if self.board_points[point] > 0:
            destination: int = point - pips
            if destination < 0:
                # bear off rule
                checkers_on_higher_points: int = sum(
                    self.player_home()[point + 1:]
                )
                if destination == -1 or checkers_on_higher_points == 0:
                    return self.apply_move(point, None), None
            elif self.board_points[destination] >= -1:
                return self.apply_move(point, destination), destination
        return None, None
    
    def move(
        self,
        point: int,
        pips: int,
    ) -> Tuple[Optional["Position"], Optional[int]]:
        """
        Try to move a checker.

        Args:
            point (int): A point of the checker before moving.
            pips (int): pips

        Returns:
            Optional[new Position]: Result of applying the move. If not applicable, return None.
            destination (Optional[int]): A point of the checker after moving. If not applicable, return None.
        """

        if self.board_points[point] > 0:
            destination: int = point - pips
            if destination >= 0 and self.board_points[destination] >= -1:
                return self.apply_move(point, destination), destination
        return None, None

    def apply_move(
        self, 
        source: Optional[int], 
        destination: Optional[int]
    ) -> "Position":
        """
        Apply a move and return a new position.

        Args:
            source (Optional[int]): A point of the checker before moving.
            destination (Optional[int]): A point of the checker after moving.

        Returns:
            Position(board_points copied): Result of applying the move.
        """
        board_points: np.ndarray = self.board_points.copy()
        player_bar: int = self.player_bar
        player_off: int = self.player_off
        opponent_bar: int = self.opponent_bar
        opponent_off: int = self.opponent_off

        if source == None:
            player_bar -= 1
        else:
            board_points[source] -= 1
        
        if destination == None:
            player_off += 1
        else:
            hit: bool = board_points[destination] == -1
            if hit:
                board_points[destination] = 1
                opponent_bar += 1
            else:
                board_points[destination] += 1
        
        return Position(
            board_points, 
            player_bar, 
            player_off, 
            opponent_bar, 
            opponent_off
        )
    
    def swap_players(self) -> "Position":
        """
        Return inverted position

        Returns:
            Position(board_points uncopied): inverted position
        """
        return Position(
            self.board_points[::-1] * -1,
            self.opponent_bar,
            self.opponent_off,
            self.player_bar,
            self.player_off,
        )

PositionType = Position