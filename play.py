"""
This file allows the player to play against the CPU's trained.
"""
from numpy.core.fromnumeric import choose
from turnBoard import turnBoard
from keras.models import *
from mcts import MCTS
from agent import choose_move_gameplay

def play():
    """
    Main driver for play.py. Automatically has num of MCTS sims set in config.py (default 25). 
    Default model level is 60
    """
    s=turnBoard()
    nnet=load_model("models/model60")
    gameOver=s.isGameOver()
    mcts=MCTS()
    playerFirst="a"
    while playerFirst not in "yYnN":
        playerFirst=input("Go first (y/n)? ")
    if playerFirst=='y' or playerFirst=='Y':
        while gameOver==2:
            s.print_board()
            player=-2
            positions=s.placable_positions(2)
            positions.append(-1)
            print(positions)
            while player not in positions:
                player=int(input("Enter square to place piece on. -1 to pass: "))
            s.push(player)
            s.print_board()
            gameOver=s.isGameOver()
            if gameOver!=2:
                if gameOver==1:
                    break
            cpu=choose_move_gameplay(nnet, mcts, s)
            s.push(cpu)
            gameOver=s.isGameOver()
        if gameOver==0:
            print("draw")
        elif gameOver==1:
            if s.turn==1:
                print("CPU wins")
            else:
                print("Player wins!")
        else:
            if s.turn==1:
                print("Player wins!")
            else:
                print("CPU wins")
        return
    else:
        while gameOver==2:
            cpu=choose_move_gameplay(nnet, mcts, s)
            s.push(cpu)
            s.print_board()
            gameOver=s.isGameOver()
            if gameOver!=2:
                break
            player=-2
            positions=s.placable_positions(2)
            positions.append(-1)
            print(positions)
            while player not in positions:
                player=int(input("Enter square to place piece on. -1 to pass: "))
            s.push(player)
            s.print_board()
            gameOver=s.isGameOver()
        if gameOver==0:
            print("draw")
        elif gameOver==1:
            if s.turn==1:
                print("Player wins!")
            else:
                print("CPU wins")
        else:
            if s.turn==1:
                print("CPU wins")
            else:
                print("Player wins!")
        return



if __name__=="__main__":
    play()