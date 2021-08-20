"""
Stores config info for training
"""

class TrainingConfig():
    def __init__(self):
        self.nnetIters=60  # number of iterations of self play for nnet (default=60)
        self.episodes=100  # number of games to train on per iteration (default=100)
        self.pitgames=20  # half the number of self play games performed after each training  (default=20)
        self.mctsSims=25  # number of mcts simulations to update policy both during episode generation and self play (default=25)
        self.c_puct=4  # c_puct for UCT (default=4)
        self.alpha=2  # alpha value for dirichlet noise (default=2)
        self.dirichletX=0.75  # percent of nnet prediction not subject to dirichlet noise (default=0.75)
    