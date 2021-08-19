"""
This file stores a Monte-Carlo Tree Search class
"""

from scipy.stats import dirichlet
import math
import tensorflow as tf
import numpy as np
from config import TrainingConfig


class MCTS:
    def __init__(self):
        self.Q={}#Array for some given state as to the rewards for taking each action
        self.N={}#Array for some given state as to the number of times each action has been visited from state
        self.P={}#Policy vector for given state
    
    def search(self, s, nnet):
        """
        Search the tree given a board state s, and neural net nnet
        """
        tc=TrainingConfig()
        gameOver=s.isGameOver()
        if gameOver!=2:  # Is game over?
            return -gameOver
        sh=s.board.tobytes()  # Compute hash
        if sh not in self.P.keys():  # Not visited->get initial nnet prediction
            prediction=nnet.predict(tf.convert_to_tensor(np.reshape(s.board,(1,8,8),order='F')))
            self.P[sh]=prediction[0][0]
            v=prediction[1][0][0]
            #Add dirichlet noise
            
            self.P[sh]=np.array(self.P[sh])
            x=tc.dirichletX
            self.P[sh]=x*self.P[sh]+(1-x)*dirichlet.rvs(np.ones(len(self.P[sh]))*tc.alpha)[0]
            self.N[sh]=np.zeros(64)
            self.Q[sh]=np.zeros(64)
            return -v

        #Find best square
        max_u, best_square = -float("inf"), -1
        for square in s.placable_positions(s.turn):
            u=self.Q[sh][square]+tc.c_puct*self.P[sh][square]*math.sqrt(sum(self.N[sh]))/(1+self.N[sh][square])
            if u>max_u:
                max_u=u
                best_square=square
        square=best_square
        sp=s.copy()
        sp.push(square)
        v=self.search(sp, nnet)
        #Recurse, update parameters
        self.Q[sh][best_square]=(self.N[sh][best_square]*self.Q[sh][best_square]+v)/(self.N[sh][best_square]+1)
        self.N[sh][best_square]+=1
        return -1