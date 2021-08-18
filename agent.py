from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay
import keras.backend as K
from nnet import ReversiModel
import numpy as np
from mcts import MCTS
from keras.models import *
from nnet import executeEpisode
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from config import TrainingConfig
from turnBoard import turnBoard


def policyIterSP(nnetIterStart=0):
    tc=TrainingConfig()
    learning_rate_fn = PiecewiseConstantDecay(boundaries=[20000, 40000, 50000], values=[1e-4, 5e-5, 2.5e-4, 1e-5])
    if nnetIterStart==0:
        nnet=ReversiModel().model
        nnet.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(learning_rate_fn))
        loss=[]
        value_out_loss=[]
        policy_out_loss=[]
    else:
        print("Loading...")
        nnet=load_model("models/model"+str(nnetIterStart))
        with open('models/model'+str(nnetIterStart)+'/loss.npy', 'rb') as f:
            loss=np.load(f)
            value_out_loss=np.load(f)
            policy_out_loss=np.load(f)
            iters=np.load(f)
        K.set_value(nnet.optimizer.iterations, iters)
    for nnetIter in range(nnetIterStart, tc.nnetIters):
        for ep in range(tc.episodes):
            episode=np.array(executeEpisode(nnet), dtype='O')
            print("ep: "+str(ep))
            if ep==0:
                examples=episode
            else:
                examples=np.concatenate((examples, episode))
        new_nnet=clone_model(nnet)
        new_nnet.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(learning_rate_fn))
        K.set_value(new_nnet.optimizer.iterations, nnet.optimizer.iterations.numpy())
        np.random.shuffle(examples)
        hist=modelFit(examples,new_nnet)
        loss=np.append(loss,hist.history['loss'])
        value_out_loss=np.append(value_out_loss,hist.history['value_out_loss'])
        policy_out_loss=np.append(value_out_loss,hist.history['policy_out_loss'])
        frac_win=pit(new_nnet, nnet)
        print(frac_win, nnetIter)
        if frac_win>=0.55:
            nnet=clone_model(new_nnet)
            nnet.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(learning_rate_fn))
            K.set_value(nnet.optimizer.iterations, new_nnet.optimizer.iterations.numpy())
        save_model(nnet, "models/model"+str(nnetIter+1))
        with open('models/model'+str(nnetIter+1)+'/loss.npy', 'wb') as f:
            np.save(f, loss)
            np.save(f, value_out_loss)
            np.save(f, policy_out_loss)
            np.save(f, nnet.optimizer.iterations.numpy())
            
            
def modelFit(examples,nnet):
    a=[]
    b=[]
    c=[]
    for i in range(len(examples)):
        a.append(np.reshape(examples[i][0],(8,8),order='F'))
        b.append(examples[i][1])
        c.append(examples[i][2])
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    return nnet.fit(a,[b,c], epochs=10, batch_size=64, verbose=0)


def choose_move_gameplay(nnet, mcts, s):
    for _ in range(TrainingConfig().mctsSims):
        mcts.search(s,nnet)
    pi=mcts.P[s.board.tobytes()]
    legalmoves=s.placable_positions(s.turn)
    if len(legalmoves)==0:
        a=-1
    else:
        legalprobs=np.take(pi,legalmoves)
        a=legalmoves[np.argmax(legalprobs)]
    return a


def pit(nnet, new_nnet):
    record=[0,0,0]#new_nnet wins, nnet wins, draws
    for i in range(TrainingConfig().pitgames):
        print("pit: "+str(i))
        s=turnBoard()
        mcts=MCTS()
        new_mcts=MCTS()
        while True:
            a=choose_move_gameplay(nnet, mcts, s)
            s.push(a)
            gameOver=s.isGameOver()
            if gameOver!=2:
                if gameOver==1:
                    record[0]+=1
                elif gameOver==-1:
                    record[1]+=1
                else:
                    record[2]+=1
                break
            a=choose_move_gameplay(new_nnet, new_mcts, s)
            s.push(a)
            gameOver=s.isGameOver()
            if gameOver!=2:
                if gameOver==1:
                    record[1]+=1
                elif gameOver==-1:
                    record[0]+=1
                else:
                    record[2]+=1
                break
        s=turnBoard()
        mcts=MCTS()
        new_mcts=MCTS()
        while True:
            a=choose_move_gameplay(new_nnet, new_mcts, s)
            s.push(a)
            gameOver=s.isGameOver()
            if gameOver!=2:
                if gameOver==1:
                    record[1]+=1
                elif gameOver==-1:
                    record[0]+=1
                else:
                    record[2]+=1
                break
            a=choose_move_gameplay(nnet, mcts, s)
            s.push(a)
            gameOver=s.isGameOver()
            if gameOver!=2:
                if gameOver==1:
                    record[0]+=1
                elif gameOver==-1:
                    record[1]+=1
                else:
                    record[2]+=1
                break
    return (record[0]+0.5*record[2])/sum(record)

if __name__=="__main__":
    policyIterSP(1)