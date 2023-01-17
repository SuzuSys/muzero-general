import numpy as np

from .config import CHECKERS, POINTS, HOME_POINTS, HEIGHT_POINTS
from .position import PositionType
from .structs import DiceType, UsedDiceType, PlayerType, NumpyType

BAR = 3
HOME = 3 * HOME_POINTS
CENTER = 3 * (POINTS//2 - HOME_POINTS)
RENDER_WIDTH = 7 + (BAR + HOME + CENTER)*2 + 1
RENDER_HEIGHT = 7 + HEIGHT_POINTS
P: NumpyType = np.array([
    0, 
    BAR+1, 
    BAR+HOME+2, 
    BAR+HOME+CENTER+3, 
    BAR+HOME+CENTER*2+4, 
    BAR+HOME*2+CENTER*2+5, 
    BAR*2+HOME*2+CENTER*2+6], dtype='int8')

class Render:
    def __init__(self):
        d: NumpyType = np.full((RENDER_HEIGHT, RENDER_WIDTH), fill_value=' ', dtype='str')
        d[[0,HEIGHT_POINTS+1,HEIGHT_POINTS+6],:] = '-'
        d[:,-1] = '\n'
        for i in P:
            self.deploy_string(d[0:HEIGHT_POINTS+2, i], "+|||+")
        self.deploy_string(d[0,P[0]+1:P[0]+1+3], "BAR")
        self.deploy_string(d[0,P[5]+1:P[5]+1+3], "BAR")
        self.deploy_string(d[0,P[1]+2:P[1]+2+4], "HOME")
        self.deploy_string(d[0,P[5]-5:P[5]-5+4], "HOME")
        n: int = 0
        for k in range(1,5):
            for j in range(P[k]+2, P[k+1], 3):
                d[HEIGHT_POINTS+1,j] = str(n%10)
                if n//10 > 0:
                    d[HEIGHT_POINTS+1,j-1] = str(n//10)
                n += 1
        self.deploy_string(d[HEIGHT_POINTS+3,:5], "DICE:")
        self.deploy_string(d[HEIGHT_POINTS+4,:14], "MOVED: [     ]")
        self.deploy_string(d[HEIGHT_POINTS+5,:7], "PLAYER:")
        self.D = d

        materials: NumpyType = np.full(CHECKERS+1, fill_value='', dtype='str')
        materials[:3] = [' ', '0', 'X']
        materials[4:] = np.arange(4,CHECKERS+1) % 10
        triO: NumpyType = np.fliplr(np.tri(CHECKERS+1, HEIGHT_POINTS, -1, dtype='int8'))
        triX: NumpyType = triO.copy() * 2
        triO[4:,0] = np.arange(4,CHECKERS+1)
        triX[4:,0] = np.arange(4,CHECKERS+1)
        allocation = np.concatenate((triO, np.flipud(triX)), axis=0)
        self.CHECKERS_DISPLAY = materials[allocation][:-1]

    def deploy_string(_, nparr: NumpyType, string: str) -> None:
        """
        Place a string in a char array
        Args:
            nparr (NumpyType): char array section
            string (str): string
        """
        for i, s in enumerate(string):
            nparr[i] = s

    def render(
        self,
        position: PositionType,
        dice: DiceType,
        used_dice: UsedDiceType,
        player: PlayerType
    ) -> str:
        """
        render.

        Args:
            position (PositionType): position
            dice (DiceType): dice
            used_dice (UsedDiceType): used_dice

        Returns:
            str: string
        """
        n = 0
        for k in range(1,5):
            for j in range(P[k]+2, P[k+1], 3):
                c = position.board_points[n]
                self.D[1:HEIGHT_POINTS+1,j] = self.CHECKERS_DISPLAY[c]
                self.D[1,j-1] = str(c//10) if c//10 > 0 else ' '
                n += 1
        self.D[1:HEIGHT_POINTS+1,P[0]+2] = self.CHECKERS_DISPLAY[-position.opponent_bar]
        self.D[1:HEIGHT_POINTS+1,P[5]+2] = self.CHECKERS_DISPLAY[position.player_bar]
        for i,bar in [(P[0], position.opponent_bar), (P[5], position.player_bar)]:
            self.D[1,i+1] = bar//10 if bar//10 > 0 else ' '
        for i,off in [(P[0], position.player_off), (P[5], position.opponent_off)]:
            self.D[HEIGHT_POINTS+2,i+2] = off%10
            self.D[HEIGHT_POINTS+2,i+1] = off//10 if off//10 > 0 else ' '
        self.D[HEIGHT_POINTS+3,6] = str(dice.left)
        self.D[HEIGHT_POINTS+3,8] = str(dice.right)
        for i in range(3):
            p = ' '
            if len(used_dice.used) > i:
                p = str(used_dice.used[i]) if used_dice.used[i] != 0 else '*'
            self.D[HEIGHT_POINTS+4,8+i*2] = p
        self.D[HEIGHT_POINTS+5,8] = str(int(player))
        return "".join(list(self.D.flatten()))

RenderType = Render