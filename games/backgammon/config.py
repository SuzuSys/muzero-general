import numpy as np

CHECKERS = 8
POINTS = 12
HOME_POINTS = POINTS // 4
PIPS_COUNT = 3 # <= HOME_POINTS

# Player_0's position
INITIAL_POSITION = np.array(
    [-2, 0, 3, 0, 0, -3, 3, 0, 0, -3, 0, 2],
    dtype='int8')
# Player_0: right to left
# Player_1: left to right

ACTION_SPACE = 1 + (POINTS + 2) * 2
# [skip|bar|POINTS|bar|bar|POINTS|bar]
#       (left die)     (right die)

# observation width/height/channels
OBS_WIDTH = POINTS + 2 # [b/o][points][o/b]
OBS_HEIGHT = CHECKERS # >= CHECKERS
OBS_DEPTH = 12
# 0: player_0's checkers (quantity)                       w.h
# 1: player_1's checkers (quantity)                       w.h
# 2: off/bar [1][0,..,0][1]                               w
# 3: player_0's home [0][0,..,0][1,..,1][0]               w
# 4: player_1's home [0][1,..,1][0,..,0][0]               w
# 5: source pips (left die)                               w
# 6: destination pips (left die)                          w
# 7: source pips (right die)                              w
# 8: destination pips (right die)                         w
# 9: number of usable dice (if 2: [1][1][0][0])           h
# 10: number of dice to use (if third move: [0][0][1][0]) h
# 11: turn (player_0: 0, player_1: 1)                     1

# If double, observation 8, 9 is 0

# render setting
HEIGHT_POINTS = 3