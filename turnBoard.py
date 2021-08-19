"""
This file stores the class of the board object used for Reversi
"""
import reversi_board
import collections


class turnBoard(reversi_board.ReversiBoard):
    def __init__(self, turn=2):
        self.turn=turn
        super(turnBoard, self).__init__()
    
    def changeTurn(self, turn):
        """
        Add turn object to reversi_board
        """
        #Maps 1->2 and 2->1
        return (-(turn-1)+2)
    
    def push(self, p):
        """
        Add a piece to square p, change turn
        """
        if p==-1: # pass turn
            self.turn=self.changeTurn(self.turn)
            return
        else:
            self.put_piece(p,self.turn)
            self.turn=self.changeTurn(self.turn)
            return
        
    
    def isGameOver(self):
        #Return 0 if draw, 1 if player whose turn it is wins, -1 if player whose turn it is loses, 2 if not over
        if len(self.placable_positions(1))!=0:
            return 2  # Not draw
        if len(self.placable_positions(2))!=0:
            return 2  # Not draw
        counts=collections.Counter(self.board)
        if counts[1]>counts[2]:  
            if self.turn==1:
                return 1
            else:
                return -1
        elif counts[1]<counts[2]:
            if self.turn==2:
                return 1
            else:
                return -1
        else:
            return 0
    
    def copy(self):
        boardCopy=turnBoard(self.turn)
        boardCopy.board=self.board.copy()
        return boardCopy