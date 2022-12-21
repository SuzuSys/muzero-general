from typing import List
import numpy as np

from .config import POINTS, HOME_POINTS, OBS_WIDTH, OBS_HEIGHT, OBS_DEPTH
from .position import PositionType
from .gammon_oneway import GammonOneway, GammonOnewayType
from .gammon_render import Render, RenderType
from .structs import MatchState, Player, MatchStateType, DiceType, UsedDiceType, NumpyType, PlayerType

# observation
OBS_TRI: NumpyType = np.tri(OBS_HEIGHT+1, OBS_HEIGHT, -1, dtype='int8')

class BackGammon:
    def __init__(self, seed: int, turn_zero: bool):
        """
        Initial the game.

        Args:
            seed (int): random seed
            turn_zero (bool): first player is zero
        """
        self.gammon: GammonOnewayType = GammonOneway(seed)
        self.turn: PlayerType = Player.ZERO if turn_zero else Player.ONE
        self.gammon.first_roll()
        self.gammon.generate_plays()
        self.ascii_board: RenderType = Render()
    
    def reset(self, turn_zero: bool) -> None:
        """
        Reset the game for a new game.

        Args:
            turn_zero (bool): first player is zero
        """
        self.gammon.reset()
        self.turn = Player.ZERO if turn_zero else Player.ONE
        self.gammon.first_roll()
        self.gammon.generate_plays()
    
    def invert(_, x: int, a: int, b: int) -> int:
        """
        Move x symmetrically about (a+b)/2.
        """
        return -x + a + b
    
    def get_legal_actions(self) -> List[int]:
        """
        Return an array of integers, subset of the action space.

        Returns:
            List[int]: an array of integers, subset of the action space
        """
        legal_actions: List[int] = self.gammon.get_legal_actions()
        if legal_actions != [0]:
            if self.turn == Player.ZERO:
                for (i, action) in enumerate(legal_actions):
                    # in  [skip]   [POINTS][bar]   [POINTS][bar]
                    # out [skip][/][POINTS][bar][/][POINTS][bar]
                    # [0]   [1-12][13]   [14-25][26]
                    action += 1
                    # [0][/][2-13][14]   [15-26][27]
                    if action > POINTS + 2:
                        action += 1
                    # [0][/][2-13][14][/][16-27][28]
                    legal_actions[i] = action
            else:
                for (i, action) in enumerate(legal_actions):
                    # in  [skip][POINTS][bar][POINTS][bar]
                    # out [skip][bar][POINTS][bar][POINTS]
                    if action <= POINTS + 1:
                        action = self.invert(
                            action, 1, POINTS + 1
                        )
                    else:
                        action = self.invert(
                            action, POINTS + 2, (POINTS + 1) * 2
                        )
                    # in  [skip][bar][POINTS]   [bar][POINTS]
                    # out [skip][bar][POINTS][/][bar][POINTS][/]
                    # [0][1][2-13]   [14][15-26]
                    if action > POINTS + 1:
                        action += 1
                    # [0][1][2-13][/][15][16-27][/]
                    legal_actions[i] = action


        return legal_actions
    
    def action(self, action: int) -> bool:
        """
        Apply action to the game.

        Args:
            action (int): action of the action_space to take.

        Returns:
            bool: game has ended
        """
        if action > 0:
            if self.turn == Player.ZERO:
                # in  [skip][/][POINTS][bar][/][POINTS][bar]
                # out [skip]   [POINTS][bar]   [POINTS][bar]
                # [0][/][2-13][14][/][16-27][28]
                if action > POINTS + 3:
                    action -= 1
                # [0][/][2-13][14]   [15-26][27]
                action -= 1
                # [0]   [1-12][13]   [14-25][26]
            else:
                # in  [skip][bar][POINTS][/][bar][POINTS][/]
                # out [skip][bar][POINTS]   [bar][POINTS]
                # [0][1][2-13][/][15][16-27][/]
                if action > POINTS + 2:
                    action -= 1
                # [0][1][2-13]   [14][15-26]

                # in  [skip][bar][POINTS][bar][POINTS]
                # out [skip][POINTS][bar][POINTS][bar]
                if action <= POINTS + 1:
                    action = self.invert(
                        action, 1, POINTS + 1
                    )
                else:
                    action = self.invert(
                        action, POINTS + 2, (POINTS + 1) * 2
                    )

        match_state: MatchStateType = self.gammon.action(action)
        if match_state == MatchState.NOTHING:
            return False
        if match_state == MatchState.TURN_OVER:
            self.gammon.swap_players()
            self.gammon.roll()
            self.gammon.generate_plays()
            self.turn = Player.ONE if self.turn == Player.ZERO else Player.ZERO
            return False
        if match_state == MatchState.GAME_OVER:
            return True
    
    def get_observation(self) -> NumpyType:
        """
        Return the game observation.

        Returns:
            NumpyType: shape: (OBS_DEPTH, OBS_WIDTH, OBS_HEIGHT)
        """
        position: PositionType = self.gammon.position
        if self.turn == Player.ONE:
            position = position.swap_players()
        # player_0's checkers move high index to low index
        # player_1's checkers move low index to high index
        w: int = OBS_WIDTH
        h: int = OBS_HEIGHT
        d: int = OBS_DEPTH
        observation: NumpyType = np.zeros((d, w, h), dtype='int8')
        # player_0 board
        player_0_board: NumpyType = np.zeros(w, dtype='int8')
        player_0_board[0] = position.player_off
        player_0_board[1:-1] = position.board_points
        player_0_board[-1] = position.player_bar
        player_0_board[player_0_board < 0] = 0
        observation[0] = OBS_TRI[player_0_board]
        # player_1 board
        player_1_board: NumpyType = player_0_board
        player_1_board[0] = position.opponent_bar
        player_1_board[1:-1] = -position.board_points
        player_1_board[-1] = position.opponent_off
        player_1_board[player_1_board < 0] = 0
        observation[1] = OBS_TRI[player_1_board]
        # off bar board
        observation[2, [0,-1]] = 1
        # player_0 home board
        home: int = HOME_POINTS
        observation[3, 1:home+1] = 1
        # player_1 home board
        observation[4, -(home+1):-1] = 1
        # left dice board
        dice: DiceType = self.gammon.dice
        observation[5, :, dice.left-1] = 1
        # right dice board
        observation[6, :, dice.right-1] = 1
        # used dice board
        used_dice: UsedDiceType = self.gammon.used_dice
        if used_dice.left > 0:
            observation[7, :, used_dice.left-1] = 1
        elif used_dice.right > 0:
            observation[7, :, -1] = 1
        # turn board
        if self.turn == Player.ONE:
            observation[8] = 1

        return observation
    
    def __str__(self):
        position: PositionType = self.gammon.position
        if self.turn == Player.ONE:
            position = position.swap_players()
        return self.ascii_board.render(
            position,
            self.gammon.dice,
            self.gammon.used_dice,
            self.turn)
