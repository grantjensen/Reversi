"""
This file stores code related to the neural net used.
"""

from keras.models import *
from keras.layers import *
from keras.optimizers import *
import numpy as np
from mcts import MCTS
from turnBoard import turnBoard
from config import TrainingConfig

class ReversiModel:
    """
    Class for Reversi neural net
    """
    def __init__(self):
        inx = x=Input((8,8))
        x=Reshape((8,8,1))(x)
        for _ in range(10):  # Create residual layers
            x=Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format='channels_last')(x)
            x=BatchNormalization(axis=3)(x)
            x=Activation("relu")(x)
            
        res_out=x
        # Policy output
        x=Conv2D(filters=2, kernel_size=1, data_format='channels_last')(res_out)
        x=BatchNormalization(axis=3)(x)
        x=Activation("relu")(x)
        x=Flatten()(x)
        policy_out=Dense(8*8, activation="softmax", name="policy_out")(x)
        self.model=policy_out
        
        #Value output
        x=Conv2D(filters=1, kernel_size=1, data_format="channels_last")(res_out)
        x=BatchNormalization(axis=3)(x)
        x=Activation("relu")(x)
        x=Flatten()(x)
        value_out=Dense(1, activation='tanh', name='value_out')(x)
        self.model=Model(inx, [policy_out, value_out], name='reversi_model')
        
        
def executeEpisode(nnet):
    """
    Generates a game of self play from nnet. Nnet still receives policy improvements from mcts.
    Returns examples: [board_state, policy, value]
    """
    searchtime=0
    examples=[]
    s=turnBoard()
    mcts=MCTS()
    move=0
    while True:
        for _ in range(TrainingConfig().mctsSims):
            mcts.search(s,nnet)
        pi=mcts.P[s.board.tobytes()]
        examples.append([s.board, pi, None])
        legalmoves=s.placable_positions(s.turn)
        if len(legalmoves)==0:
            a=-1
        else:
            legalprobs=np.take(pi,legalmoves)
            if move<30: #Temperature
                legalprobs/=sum(legalprobs)
                a=np.random.choice(legalmoves, p=legalprobs)
            else:
                a=legalmoves[np.argmax(legalprobs)]
        s.push(a)
        gameover=s.isGameOver()
        if gameover!=2:
            return assignRewards(examples, gameover)
        

def assignRewards(examples, reward):
    """
    Back propogates the final rewared of the game to the examples
    """
    for i in range(len(examples)-1,-1,-1):
        examples[i][2]=reward
        reward*=-1
    
    return examples