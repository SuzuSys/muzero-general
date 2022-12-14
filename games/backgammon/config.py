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

ACTION_SPACE = 1 + (POINTS + 1) * 2
# [skip|POINTS|bar|POINTS|bar]
#       (left die) (right die)

# observation width/height/channels
OBS_WIDTH = POINTS + 2 # [b/o][points][o/b]
OBS_HEIGHT = CHECKERS # >= CHECKERS
OBS_DEPTH = 9
# 1: player_0's checkers (quantity)
# 2: player_1's checkers (quantity)
# 3: off/bar [1][0,..,0][1]
# 4: player_0's home [0][0,..,0][1,..,1][0]
# 5: player_1's home [0][1,..,1][0,..,0][0]
# 6: left dice (numerical)
# 7: right dice (numerical)
# 8: chose dice (left dice: downer, right dice: upper)
# 9: turn (player_0: 0, player_1: 1)

# render setting
HEIGHT_POINTS = 3