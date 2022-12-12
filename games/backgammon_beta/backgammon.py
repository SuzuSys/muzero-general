# Copyright 2020 Softwerks LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import itertools
import operator
import random
from typing import Callable, List, NamedTuple, Optional, Tuple, Set
import numpy

from .match import Match, Player, GameState
from .match import decode as match_decode
from .position import Position
from .position import decode as position_decode

import collections

MatchType = Match
PositionType = Position

STARTING_POSITION_ID = "4HPwATDgc/ABMA"
STARTING_MATCH_ID = "cAgAAAAAAAAA"

CHECKERS = 15
POINTS = 24
POINTS_PER_QUADRANT = int(POINTS / 4)

ASCII_BOARD_HEIGHT = 11
ASCII_MAX_CHECKERS = 5
ASCII_13_24 = "+12-13-14-15-16-17------18-19-20-21-22-23-+"
ASCII_12_01 = "+11-10--9--8--7--6-------5--4--3--2--1--0-+"


class BackgammonError(Exception):
    pass


class MoveState(enum.Enum):
    BEAR_OFF = enum.auto()
    ENTER_FROM_BAR = enum.auto()
    DEFAULT = enum.auto()


class Move(NamedTuple):
    pips: int
    source: Optional[int]
    destination: Optional[int]


class Play(NamedTuple):
    moves: Tuple[Move, ...]
    position: PositionType


class Backgammon:
    def __init__(
        self, seed: int, position_id: str = STARTING_POSITION_ID, match_id: str = STARTING_MATCH_ID
    ):
        self.position: PositionType = position_decode(position_id)
        self.virtual_position: PositionType = self.position
        self.legal_moves: List[Play] = []
        self.move_index: int = 0
        self.used_right_dice: bool = False
        self.match: MatchType = match_decode(match_id)
        random.seed(seed)
    
    def reset(self):
        self.position: PositionType = position_decode(STARTING_POSITION_ID)
        self.virtual_position: PositionType = self.position
        self.legal_moves: List[Play] = []
        self.move_index: int = 0
        self.used_right_dice: bool = False
        self.match: MatchType = match_decode(STARTING_MATCH_ID)
    
    def output_observation(self) -> numpy.ndarray:
        # (26,6) * 9 = (9,26,6)
        player_0_board, player_1_board = self.output_board_observation()
        player_0_board_onehot = numpy.tri(7,6,-1)[player_0_board]
        player_1_board_onehot = numpy.tri(7,6,-1)[player_1_board]
        off_bar_board = numpy.zeros((26,6))
        off_bar_board[[0,25]] = numpy.ones(6)
        player_0_home_board = numpy.zeros((26,6))
        player_0_home_board[1:7] = numpy.ones(6)
        player_1_home_board = numpy.zeros((26,6))
        player_1_home_board[-7:-1] = numpy.ones(6)
        left_dice_board = numpy.identity(6)[numpy.full(26, self.match.dice[0] - 1)]
        right_dice_board = numpy.identity(6)[numpy.full(26, self.match.dice[1] - 1)]
        if self.move_index == 0:
            chose_dice_board = numpy.zeros((26,6))
        else:
            if self.used_right_dice:
                chose_dice_board = numpy.identity(6)[numpy.full(26, 5)]
            else:
                chose_dice_board = numpy.identity(6)[numpy.full(26, self.move_index - 1)]
        turn_board = numpy.zeros((26,6)) if self.match.player == Player.ZERO else numpy.ones((26,6))
        return numpy.array([
            player_0_board_onehot,
            player_1_board_onehot,
            off_bar_board,
            player_0_home_board,
            player_1_home_board,
            left_dice_board,
            right_dice_board,
            chose_dice_board,
            turn_board,
        ])

    def output_board_observation(self) -> Tuple[List[int], List[int]]:
        # 常にPlayer ZERO は左に進み、Player ONE は右に進む。
        # 出力は (Player ZERO, Player ONE)
        # [g/b][6*4][b/g]
        def classify_player(x: int) -> int:
            if x < 0:
                return 0
            elif x > 6:
                return 6
            else:
                return x
        def classify_opponent(x: int) -> int:
            if x > 0:
                return 0
            elif x < -6:
                return 6
            else:
                return -x
        player_board_points: List[int] = list(map(classify_player, self.virtual_position.board_points))
        opponent_board_points: List[int] = list(map(classify_opponent, self.virtual_position.board_points))
        player_board: List[int] = (
            [self.virtual_position.player_off if self.virtual_position.player_off <= 6 else 6]
            + player_board_points
            + [self.virtual_position.player_bar if self.virtual_position.player_bar <= 6 else 6]
        )
        opponent_board: List[int] = (
            [self.virtual_position.opponent_bar if self.virtual_position.player_bar <= 6 else 6]
            + opponent_board_points
            + [self.virtual_position.opponent_off if self.virtual_position.opponent_off <= 6 else 6]
        )
        if self.match.player == Player.ZERO:
            zero_board, one_board = player_board, opponent_board
        else: # self.match.player == Player.ONE
            zero_board, one_board = opponent_board[::-1], player_board[::-1]
        return (zero_board, one_board)

    def generate_plays(self) -> List[Play]:
        """Generate and return legal plays."""

        # サイコロの数を2個に限定
        def generate(
            position: PositionType,
            dice: Tuple[int, ...],
            die: int = 0, # 最初にどちらのサイコロを選ぶか
            moves: Tuple[Move, ...] = (),
            plays: List[Play] = [],
        ) -> List[Play]:
            """Generate and return all plays."""
            new_position: Optional[PositionType]
            destination: Optional[int]
            point: int
            num_checkers: int
            pips: int

            if die < len(dice):
                pips = dice[die]

                if position.player_bar > 0:
                    new_position, destination = position.enter(pips)
                    if new_position:
                        generate(
                            new_position,
                            dice,
                            die + 1,
                            moves + (Move(pips, None, destination),),
                            plays,
                        )
                elif sum(position.player_home()) + position.player_off == CHECKERS:
                    for point, num_checkers in enumerate(
                        position.board_points[:POINTS_PER_QUADRANT]
                    ):
                        new_position, destination = position.off(point, pips)
                        if new_position:
                            generate(
                                new_position,
                                dice,
                                die + 1,
                                moves + (Move(pips, point, destination),),
                                plays,
                            )
                else:
                    for point, num_checkers in enumerate(position.board_points):
                        new_position, destination = position.move(point, pips)
                        if new_position:
                            generate(
                                new_position,
                                dice,
                                die + 1,
                                moves + (Move(pips, point, destination),),
                                plays,
                            )

            if len(moves) > 0:
                plays.append(Play(moves, position))

            return plays

        doubles: bool = self.match.dice[0] == self.match.dice[1]
        dice: Tuple[int, ...] = self.match.dice * 2 if doubles else self.match.dice

        plays: List[Play] = generate(self.position, dice, plays=[])
        if not doubles:
            temp_plays = generate(self.position, dice[::-1], plays=[])
            plays += temp_plays
        # 重複あり

        if plays:
            max_moves: int = max(len(p.moves) for p in plays)
            if max_moves == 1:
                max_pips: int = max(dice)
                higher_plays: List[Play] = list(
                    filter(lambda p: p.moves[0].pips == max_pips, plays)
                )
                if higher_plays:
                    plays = higher_plays
            else:
                # movesの短いものは消去
                plays = list(filter(lambda p: len(p.moves) == max_moves, plays))
            
            # key_func: Callable = lambda p: hash(p.position)
            # plays = sorted(plays, key=key_func)

            # plays = list(
            #    map(
            #        next,
            #        map(operator.itemgetter(1), itertools.groupby(plays, key_func)),
            #    )
            #)


        return plays
    
    def generate_legal_moves(self): # -> legal_moves
        """Generate and return legal plays."""

        # サイコロの数を2個に限定
        def generate(
            position: PositionType,
            dice: Tuple[int, ...],
            die: int = 0, # 最初にどちらのサイコロを選ぶか
            moves: Tuple[Move, ...] = (),
            plays: List[Play] = [],
        ) -> List[Play]:
            """Generate and return all plays."""
            new_position: Optional[PositionType]
            destination: Optional[int]
            point: int
            num_checkers: int
            pips: int

            if die < len(dice):
                pips = dice[die]

                if position.player_bar > 0:
                    new_position, destination = position.enter(pips)
                    if new_position:
                        generate(
                            new_position,
                            dice,
                            die + 1,
                            moves + (Move(pips, None, destination),),
                            plays,
                        )
                elif sum(position.player_home()) + position.player_off == CHECKERS:
                    for point, num_checkers in enumerate(
                        position.board_points[:POINTS_PER_QUADRANT]
                    ):
                        new_position, destination = position.off(point, pips)
                        if new_position:
                            generate(
                                new_position,
                                dice,
                                die + 1,
                                moves + (Move(pips, point, destination),),
                                plays,
                            )
                else:
                    for point, num_checkers in enumerate(position.board_points):
                        new_position, destination = position.move(point, pips)
                        if new_position:
                            generate(
                                new_position,
                                dice,
                                die + 1,
                                moves + (Move(pips, point, destination),),
                                plays,
                            )

            if len(moves) > 0:
                plays.append(Play(moves, position))

            return plays

        doubles: bool = self.match.dice[0] == self.match.dice[1]
        dice: Tuple[int, ...] = self.match.dice * 2 if doubles else self.match.dice

        plays: List[Play] = generate(self.position, dice, plays=[])
        if not doubles:
            plays += generate(self.position, dice[::-1], plays=[])
        # 重複あり
        
        if plays:
            max_moves: int = max(len(p.moves) for p in plays)
            if max_moves == 1:
                max_pips: int = max(dice)
                higher_plays: List[Play] = list(
                    filter(lambda p: p.moves[0].pips == max_pips, plays)
                )
                if higher_plays:
                    plays = higher_plays
            else:
                # movesの短いものは消去
                plays = list(filter(lambda p: len(p.moves) == max_moves, plays))
        self.legal_moves = plays

    def move_filter(self, is_right_dice: bool, source: Optional[int]): # legal_moves -> legal_moves, virtual_position, move_index, 
        # source に None あり!!!
        pips: int = self.match.dice[is_right_dice]

        temp_legal_moves = list(filter(
            lambda p: (p.moves[self.move_index].pips == pips) and (p.moves[self.move_index].source == source), 
            self.legal_moves
        ))
        if len(temp_legal_moves) == 0:
            # bug!!!!!
            print('move_filter')
            print(f'dice: {self.match.dice}')
            print(f'pips: {pips}')
            print(f'self.legal_moves: {self.legal_moves}')
            print(f'temp_legal_moves: {temp_legal_moves}')
        self.legal_moves = temp_legal_moves
        if source == None:
            self.virtual_position, _ = self.virtual_position.enter(pips)
        else:
            self.virtual_position = self.virtual_position.apply_move(source, source - pips if source - pips > -1 else None)
        
        self.move_index += 1
        doubles: bool = self.match.dice[0] == self.match.dice[1]
        if not doubles:
            self.used_right_dice = is_right_dice
    
    def output_legal_actions(self) -> List[int]:
        # 外部によって len(self.legal_moves[0].moves) <= self.move_index に当てはまらないことが条件
        # sources に None あり!!!
        # 1    + (6*4+1)       + (6*4+1)
        # skip + left_dice,pos + right_dice,pos
        if len(self.legal_moves) == 0:
            return [0]
        else:
            doubles: bool = self.match.dice[0] == self.match.dice[1]
            if doubles:
                # source の選択肢を全て採る
                if self.legal_moves[0].moves[self.move_index].source == None:
                    # source が bar
                    if self.match.player == Player.ONE:
                        return [1]
                    else: # self.match.player == Player.ZERO
                        return [1 + POINTS]
                else:
                    legal_sources_set :Set[int] = set()
                    legal_sources :List[int] = []
                    for play in self.legal_moves:
                        legal_sources_set.add(play.moves[self.move_index].source)
                    if self.match.player == Player.ONE:
                        legal_sources = list(map(lambda p: 2 + (POINTS-1 - p), legal_sources_set))
                    else: # self.match.player == Player.ZERO
                        legal_sources = list(map(lambda p: 1 + p, legal_sources_set))
                    legal_sources.sort()
                    return legal_sources
            else:
                legal_sources :List[int] = []
                can_use_right = False
                can_use_left = False
                for i in self.legal_moves:
                    pips = i.moves[self.move_index].pips
                    can_use_right = pips == self.match.dice[1]
                    can_use_left = pips == self.match.dice[0]
                    if can_use_right and can_use_left:
                        break

                if can_use_left:
                    # used right dice
                    # We'll use left dice
                    if self.legal_moves[0].moves[self.move_index].source == None:
                        if self.match.player == Player.ONE:
                            legal_sources.append(1)
                        else: # self.match.player == Player.ZERO
                            legal_sources.append(1 + POINTS)
                    else:
                        legal_sources_set :Set[int] = set()
                        for play in self.legal_moves:
                            if play.moves[self.move_index].pips == self.match.dice[0]:
                                legal_sources_set.add(play.moves[self.move_index].source)
                        if self.match.player == Player.ONE:
                            legal_sources += list(map(lambda p: 2 + (POINTS-1 - p), legal_sources_set))
                        else: # self.match.player == Player.ZERO
                            legal_sources += list(map(lambda p: 1 + p, legal_sources_set))
                    
                if can_use_right:
                    # used left dice
                    # We'll use right dice
                    if self.legal_moves[0].moves[self.move_index].source == None:
                        if self.match.player == Player.ONE:
                            legal_sources.append(2 + POINTS)
                        else: # self.match.player == Player.ZERO
                            legal_sources.append(1 + POINTS + 1 + POINTS)
                    else:
                        legal_sources_set :Set[int] = set()
                        for play in self.legal_moves:
                            if play.moves[self.move_index].pips == self.match.dice[1]:
                                legal_sources_set.add(play.moves[self.move_index].source)
                        if self.match.player == Player.ONE:
                            legal_sources += list(map(lambda p: 2 + POINTS + 1 + (POINTS-1 - p), legal_sources_set))
                        else: # self.match.player == Player.ZERO
                            legal_sources += list(map(lambda p: 1 + POINTS + 1 + p, legal_sources_set))
                legal_sources.sort()
                return legal_sources
    def classify_action(self, action: int) -> Tuple[bool, Optional[int]]:
        # action != 0
        is_right_dice: bool = None
        source: Optional[int] = None
        if self.match.player == Player.ZERO:
            if action < 1 + POINTS:
                # 0 < action < 1 + POINTS
                is_right_dice = False
                source = action - 1
            elif action < 1 + POINTS + 1:
                # 1 + POINTS <= action < 1 + POINTS + 1
                is_right_dice = False
                source = None
            elif action < 1 + POINTS + 1 + POINTS:
                # 1 + POINTS + 1 <= action < 1 + POINTS + 1 + POINTS
                is_right_dice = True
                source = action - (1 + POINTS + 1)
            else:
                # 1 + POINTS + 1 + POINTS <= action < 1 + POINTS + 1 + POINTS + 1
                is_right_dice = True
                source = None
        else: # self.match.player == Player.ONE
            if action < 2:
                # 0 < action < 2
                is_right_dice = False
                source = None
            elif action < 2 + POINTS:
                # 2 <= action < 2 + POINTS
                is_right_dice = False
                source = (POINTS-1) - (action - 2)
            elif action < 2 + POINTS + 1:
                # 2 + POINTS <= action < 2 + POINTS + 1
                is_right_dice = True
                source = None
            else:
                # 2 + POINTS + 1 <= action < 2 + POINTS + 1 + POINTS
                is_right_dice = True
                source = (POINTS-1) - (action - (2 + POINTS + 1))
        return is_right_dice, source

    def action(self, action: int) -> bool: # is_end_turned
        """actionを適用する

        Args:
            action (int): 

        Returns:
            bool: end turned
        """
        if action == 0:
            self.skip()
            return True
        else:
            is_right_dice, source = self.classify_action(action)
            debug_legal_moves = self.legal_moves
            self.move_filter(is_right_dice, source) # 
            if len(self.legal_moves) == 0:
                print('BUG!!!')
                print(f'debug_legal_moves: {debug_legal_moves}')
                print(f'self.move_index: {self.move_index}')
                print(f'action: {action}')
                print(f'is_right_dice: {is_right_dice}')
                print(f'source: {source}')
            if len(self.legal_moves[0].moves) <= self.move_index: # bug: index out of range
                # 最後のactionを実行した
                # len(legal_moves) == 1 になっているはず
                all_moves :Tuple[Move, ...] = self.legal_moves[0].moves
                all_moves_tuple: Tuple[Tuple[Optional[int], Optional[int]], ...] = ()
                for move in all_moves:
                    all_moves_tuple = all_moves_tuple + ((move.source, move.destination), )
                self.play(all_moves_tuple)
                return True
            else:
                return False
            

    def start(self) -> "Backgammon":
        self.match.game_state = GameState.PLAYING
        self.match.length = 1
        self.first_roll()

        return self

    def roll(self) -> Tuple[int, int]:
        if self.match.dice != (0, 0):
            raise BackgammonError(f"Dice have already been rolled: {self.match.dice}")

        self.match.dice = (
            random.randrange(1, 6),
            random.randrange(1, 6),
        )
        return self.match.dice

    def first_roll(self) -> Tuple[int, int]:
        while True:
            self.match.dice = (
                random.randrange(1, 6),
                random.randrange(1, 6),
            )
            if self.match.dice[0] != self.match.dice[1]:
                break
        
        #if self.match.dice[0] > self.match.dice[1]:
        #    self.match.player = Player.ZERO
        #    self.match.turn = Player.ZERO
        #else:
        #    self.match.player = Player.ONE
        #    self.match.turn = Player.ONE

        # CHANGED -----------------------------------------------
        self.match.player = Player.ZERO
        self.match.turn = Player.ZERO
        # -------------------------------------------------------

        return self.match.dice

    def play(
        self, moves: Tuple[Tuple[Optional[int], Optional[int]], ...]
    ) -> "Backgammon":
        """Excecute a play, a sequence of moves."""
        new_position: PositionType = self.position
        for source, destination in moves:
            new_position = new_position.apply_move(source, destination)

        legal_plays: List[Play] = self.generate_plays()

        if new_position in [play.position for play in legal_plays]:
            self.position = new_position

            if self.position.player_off == CHECKERS:
                self.match.update_score()
            else:
                self.end_turn()

        else:
            position_id: str = self.position.encode()
            match_id: str = self.match.encode()
            raise BackgammonError(f"Invalid move: {position_id}:{match_id} {moves}")

        return self

    def skip(self) -> "Backgammon":
        num_plays: int = len(self.generate_plays())
        if num_plays == 0:
            self.end_turn()
        else:
            raise BackgammonError(f"Cannot skip turn: {num_plays} possible plays")

        return self

    def end_turn(self) -> "Backgammon":
        self.position = self.position.swap_players()
        self.match.swap_players()
        self.match.reset_dice()
        # 初期化
        self.virtual_position = self.position
        self.legal_moves = []
        self.move_index = 0
        self.used_right_dice = False
        # dice
        self.roll()

        return self

    def encode(self) -> str:
        return f"{self.position.encode()}:{self.match.encode()}"

    def __repr__(self):
        position_id: str = self.position.encode()
        match_id: str = self.match.encode()
        return f"{__name__}.{self.__class__.__name__}('{position_id}', '{match_id}')"

    def __str__(self):
        def checkers(top: List[int], bottom: List[int]) -> List[List[str]]:
            """Return an ASCII checker matrix."""
            ascii_checkers: List[List[str]] = [
                ["   " for j in range(len(top))] for i in range(ASCII_BOARD_HEIGHT)
            ]

            for half in (top, bottom):
                for col, num_checkers in enumerate(half):
                    row: int = 0 if half is top else len(ascii_checkers) - 1
                    for i in range(abs(num_checkers)):
                        if (
                            abs(num_checkers) > ASCII_MAX_CHECKERS
                            and i == ASCII_MAX_CHECKERS - 1
                        ):
                            ascii_checkers[row][col] = f" {abs(num_checkers)} "
                            break
                        ascii_checkers[row][col] = " O " if num_checkers > 0 else " X "
                        row += 1 if half is top else -1

            return ascii_checkers

        def split(position: List[int]) -> Tuple[List[int], List[int]]:
            """Return a position split into top (Player.ZERO 12-1) and bottom (Player.ZERO 13-24) halves."""

            def normalize(position: List[int]) -> List[int]:
                """Return position for Player.ZERO"""
                if self.match.player is Player.ONE:
                    position = list(map(lambda n: -n, position[::-1]))
                return position

            position = normalize(position)

            half_len: int = int(len(position) / 2)
            top: List[int] = position[:half_len][::-1]
            bottom: List[int] = position[half_len:]

            return top, bottom

        points: List[List[str]] = checkers(*split(self.position.board_points))

        bar: List[List[str]] = checkers(
            *split(
                [
                    self.position.player_bar,
                    -self.position.opponent_bar,
                ]
            )
        )

        ascii_board: str = ""
        position_id: str = self.position.encode()
        ascii_board += f"                 Position ID: {position_id}\n"
        match_id: str = self.match.encode()
        ascii_board += f"                 Match ID   : {match_id}\n"
        ascii_board += (
            " "
            + (ASCII_12_01 if self.match.player is Player.ZERO else ASCII_13_24)
            + "\n"
        )
        for i in range(len(points)):
            ascii_board += (
                ("^|" if self.match.player is Player.ZERO else "v|")
                if i == int(ASCII_BOARD_HEIGHT / 2)
                else " |"
            )
            ascii_board += "".join(points[i][:POINTS_PER_QUADRANT])
            ascii_board += "|"
            ascii_board += "BAR" if i == int(ASCII_BOARD_HEIGHT / 2) else bar[i][0]
            ascii_board += "|"
            ascii_board += "".join(points[i][POINTS_PER_QUADRANT:])
            ascii_board += "|"
            ascii_board += "\n"
        ascii_board += (
            " "
            + (ASCII_13_24 if self.match.player is Player.ZERO else ASCII_12_01)
            + "\n"
        )

        return ascii_board
